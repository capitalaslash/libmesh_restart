// libmesh restart example inspired by adaptivity_ex5

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <string>

#include <libmesh/dirichlet_boundaries.h>
#include <libmesh/dof_map.h>
#include <libmesh/enum_norm_type.h>
#include <libmesh/enum_xdr_mode.h>
#include <libmesh/equation_systems.h>
#include <libmesh/exodusII_io.h>
#include <libmesh/fe_type.h>
#include <libmesh/libmesh.h>
#include <libmesh/linear_implicit_system.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/quadrature_gauss.h>
#include <libmesh/replicated_mesh.h>
#include <libmesh/sparse_matrix.h>
#include <libmesh/system_norm.h>
#include <libmesh/transient_system.h>
#include <libmesh/zero_function.h>

using namespace libMesh;

// declarations
void system_assemble(EquationSystems & es, std::string const & system_name);
void system_init(EquationSystems & es, std::string const & system_name);

Real exact_solution(const Real x, const Real y, const Real /*t*/)
{
  return x * (1.0 - x) * y * (1.0 - y);
}

Number exact_value(
    const Point & p,
    const Parameters & parameters,
    const std::string &,
    const std::string &)
{
  return exact_solution(p(0), p(1), parameters.get<Real>("time"));
}

int main(int argc, char * argv[])
{
  LibMeshInit init{argc, argv};

  auto const sep = std::string(88, '=') + "\n";

  ReplicatedMesh mesh{init.comm()};
  MeshTools::Generation::build_square(mesh, 10, 10, 0.0, 1.0, 0.0, 1.0, QUAD4);

  EquationSystems equation_systems{mesh};

  TransientLinearImplicitSystem & system =
      equation_systems.add_system<TransientLinearImplicitSystem>("Poisson");

  system.add_variable("u", FIRST);

  system.attach_assemble_function(system_assemble);

  system.attach_init_function(system_init);

  DofMap & dof_map = system.get_dof_map();

  dof_map.add_dirichlet_boundary(
      DirichletBoundary{{0, 1, 2, 3}, {0}, ZeroFunction<Number>{}});

  equation_systems.init();

  equation_systems.print_info();

  equation_systems.parameters.set<Real>("diffusivity") = 0.1;
  equation_systems.parameters.set<uint>("linear solver maximum iterations") = 250U;
  equation_systems.parameters.set<Real>("linear solver convergence") = 1.e-8;

  ExodusII_IO io{mesh};
  auto const outfile = "out.e";
  io.write_equation_systems(outfile, equation_systems);

  Real const dt = 0.1;
  equation_systems.parameters.set<Real>("dt") = dt;
  system.time = 0.0;
  // exodus crashes if given 0 timestep
  io.write_timestep(outfile, equation_systems, 1, system.time);

  uint const num_steps = 10;

  for (uint t_step = 0; t_step < num_steps; ++t_step)
  {
    system.time += dt;

    equation_systems.parameters.set<Real>("time") = system.time;
    equation_systems.parameters.set<Real>("dt") = dt;

    fmt::print(
        out, "{}timestep {:3}, current time: {}\n", sep, t_step + 1, system.time);

    *system.old_local_solution = *system.current_local_solution;

    system.solve();

    Real const norm_h1 = system.calculate_norm(*system.solution, SystemNorm{H1});
    fmt::print(out, "H1 norm of the solution: {:e}\n", norm_h1);

    io.write_timestep(outfile, equation_systems, t_step + 2, system.time);
  }

  Real const u_norm = system.calculate_norm(*system.solution, SystemNorm{H1});
  fmt::print(out, "H1 norm of the solution: {:e}\n", u_norm);
  Real const uold_norm =
      system.calculate_norm(*system.old_local_solution, SystemNorm{H1});
  fmt::print(out, "H1 norm of the old solution: {:e}\n", uold_norm);

  mesh.write("saved_mesh.xdr");
  // ADDITIONAL_DATA is required to store old solutions
  equation_systems.write(
      "saved_solution.xdr",
      ENCODE,
      EquationSystems::WRITE_DATA | EquationSystems::WRITE_ADDITIONAL_DATA);

  // restart from saved state
  fmt::print(out, "{}RESTART\n{}", sep, sep);

  ReplicatedMesh mesh_restart{init.comm()};
  mesh_restart.read("saved_mesh.xdr");
  EquationSystems es_restart{mesh_restart};

  es_restart.read(
      "saved_solution.xdr",
      DECODE,
      EquationSystems::READ_HEADER | EquationSystems::READ_DATA |
          EquationSystems::READ_ADDITIONAL_DATA);

  auto & system_restart =
      es_restart.get_system<TransientLinearImplicitSystem>("Poisson");

  Real const unorm_restart =
      system_restart.calculate_norm(*system_restart.solution, SystemNorm{H1});
  fmt::print(out, "H1 norm of the solution after reading: {:e}\n", unorm_restart);
  Real const uoldnorm_restart =
      system_restart.calculate_norm(*system_restart.old_local_solution, SystemNorm{H1});
  fmt::print(
      out, "H1 norm of the old solution after reading: {:e}\n", uoldnorm_restart);

  DofMap & dof_map_restart = system_restart.get_dof_map();

  dof_map_restart.add_dirichlet_boundary(
      DirichletBoundary{{0, 1, 2, 3}, {0}, ZeroFunction<Number>{}});

  es_restart.reinit();

  es_restart.print_info();

  es_restart.parameters.set<Real>("diffusivity") =
      equation_systems.parameters.get<Real>("diffusivity");
  es_restart.parameters.set<uint>("linear solver maximum iterations") = 250U;
  es_restart.parameters.set<Real>("linear solver convergence") = 1.e-8;

  system_restart.attach_assemble_function(system_assemble);
  system_restart.time = system.time;

  ExodusII_IO io_restart{mesh_restart};
  auto const outfile_restart = "out_restart.e";
  io_restart.write_equation_systems(outfile_restart, es_restart);
  io_restart.write_timestep(outfile_restart, es_restart, 1, system_restart.time);

  for (uint t_step = 0; t_step < num_steps; ++t_step)
  {
    system_restart.time += dt;

    es_restart.parameters.set<Real>("time") = system_restart.time;
    es_restart.parameters.set<Real>("dt") = dt;

    fmt::print(
        out,
        "{}timestep {:3}, current time: {}\n",
        sep,
        t_step + 1,
        system_restart.time);

    *system_restart.old_local_solution = *system_restart.current_local_solution;

    system_restart.solve();

    Real const norm_h1 =
        system_restart.calculate_norm(*system_restart.solution, SystemNorm{H1});
    fmt::print(out, "H1 norm of the solution: {:e}\n", norm_h1);

    io_restart.write_timestep(
        outfile_restart, es_restart, t_step + 2, system_restart.time);
  }

  return 0;
}

void system_assemble(EquationSystems & es, std::string const & system_name)
{
  libmesh_assert_equal_to(system_name, "Poisson");

  MeshBase const & mesh = es.get_mesh();
  uint const dim = mesh.mesh_dimension();
  auto & system = es.get_system<TransientLinearImplicitSystem>(system_name);

  FEType fe_type = system.variable_type(0);
  std::unique_ptr<FEBase> fe{FEBase::build(dim, fe_type)};
  QGauss qrule{dim, fe_type.default_quadrature_order()};
  fe->attach_quadrature_rule(&qrule);

  std::vector<Real> const & JxW = fe->get_JxW();
  std::vector<std::vector<Real>> const & phi = fe->get_phi();
  std::vector<std::vector<RealGradient>> const & dphi = fe->get_dphi();

  auto const & dof_map = system.get_dof_map();

  DenseMatrix<Number> Ke;
  DenseVector<Number> Fe;

  std::vector<dof_id_type> dof_indices;

  Real const k = es.parameters.get<Real>("diffusivity");
  Real const dt = es.parameters.get<Real>("dt");

  SparseMatrix<Number> & matrix = system.get_system_matrix();

  for (auto const & elem: mesh.active_local_element_ptr_range())
  {
    // set up for current element
    dof_map.dof_indices(elem, dof_indices);

    fe->reinit(elem);

    uint const n_dofs = dof_indices.size();

    Ke.resize(n_dofs, n_dofs);
    Fe.resize(n_dofs);

    // fill local matrix and rhs
    for (uint qp = 0; qp < qrule.n_points(); ++qp)
    {
      Number u_old = 0.0;
      for (uint l = 0; l < n_dofs; ++l)
      {
        u_old += phi[l][qp] * system.old_solution(dof_indices[l]);
      }

      for (uint i = 0; i < n_dofs; ++i)
      {
        Fe(i) += JxW[qp] * (u_old * phi[i][qp] + dt * 1.0 * phi[i][qp]);

        for (uint j = 0; j < n_dofs; ++j)
        {
          Ke(i, j) += JxW[qp] *
                      (phi[j][qp] * phi[i][qp] + dt * k * (dphi[j][qp] * dphi[i][qp]));
        }
      }
    }

    // apply bcs
    dof_map.constrain_element_matrix_and_vector(Ke, Fe, dof_indices);

    // store local data in global matrix and rhs
    matrix.add_matrix(Ke, dof_indices);
    system.rhs->add_vector(Fe, dof_indices);
  }
}

void system_init(EquationSystems & es, std::string const & system_name)
{
  libmesh_assert_equal_to(system_name, "Poisson");

  auto & system = es.get_system<TransientLinearImplicitSystem>(system_name);

  es.parameters.set<Real>("time") = system.time = 0;

  system.project_solution(exact_value, nullptr, es.parameters);
}
