# Restart in libMesh

Check `libMesh` restart capabilities.

Based on `adaptivity_ex5`.

## requirements

`libMesh` should be configured with mpi, petsc and exodusII support.

Using `spack`:

```
spack install libmesh@master +mpi +petsc +exodusii
```

`libMesh` is found via `pkgconfig`, so make sure to have its folder available:

```
export PKG_CONFIG_PATH=$LIBMESH_DIR/lib[64]/pkgconfig
```

