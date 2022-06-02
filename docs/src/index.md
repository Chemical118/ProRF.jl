# ProRF.jl Documentation
## Overview
`ProRF` provides a full process for applying the random forest model of protein sequences using `DecisionTree`.

## Install
!!! warning
    `ProRF` uses Python module `Bokeh`, `Matplotlib` to provide UI. Please install these module or execute below code before add `ProRF`.

    ```bash
    $ pip install matplotlib
    $ pip install bokeh
    ```
    For more information, see `PyCall` [documentaion](https://github.com/JuliaPy/PyCall.jl).

```julia
using Pkg
Pkg.add("https://github.com/Chemical118/ProRF.jl")
```

## Examples
!!! tip "Performance Tip"
    `ProRF` support parallel computing, please turn on the julia with multiple threads. This can speed up execution time fairly.

    ```bash
    $ julia --threads auto
    ```
    For more information, see Multi-Threading [documentaion](https://docs.julialang.org/en/v1/manual/multi-threading/).

!!! note
    `ProRF` recommends interactive mode like `IJulia`, however if you want to run in non-interactive mode, execute below code to see graphs.
    ```julia
    julia_isinteractive(false)
    ```
    For more information, see [`julia_isinteractive`](@ref).

### Data preprocessing
`ProRF` has a useful function for preprocessing data.
```julia
using ProRF, Printf

Find, Lind = data_preprocess_index("Data/algpdata.fasta", val_mode=true)
@printf "%d %d\n" Find Lind

data_preprocess_fill(Find, Lind,
                     "Data/algpdata.fasta",
                     "Data/Mega/ealtreedata.nwk",
                     "Data/jealgpdata.fasta",
                     val_mode=true);

view_sequence("Data/jealgpdata.fasta", save=true)
```

### Find best random forest arguments
`ProRF` helps you find arguments for the random forset.
```julia
using ProRF, Printf

RI = RFI("Data/jealgpdata.fasta", "Data/data.xlsx", 2:1:10, 100:10:500)
X, Y, L = get_data(RI, 9, 'E')

MeZ, SdZ = iter_get_reg_value(RI, X, Y, 10, val_mode=true)

view_reg3d(RI, MeZ, title="NRMSE value", azim=90, scale=3)
view_reg3d(RI, SdZ, title="NRMSE SD value", elev=120, scale=3)

N_Feature, N_Tree = get_reg_value_loc(RI, MZ)
@printf "Best Arguments : %d %d\n" N_Feature N_Tree
```

### Execute random forest
`ProRF` executes random forest flexibly and easily.
```julia
using ProRF, Printf

R = RF("Data/jealgpdata.fasta", "Data/data.xlsx")
X, Y, L = get_data(R, 9, 'E', blosum=80)

M = rf_model(X, Y, N_Feature, N_Tree)
@printf "Total NRMSE : %.6f\n" nrmse(M, X, Y)

MeF, SdF = iter_get_reg_importance(R, X, Y, L, N_Feature, N_Tree, 100, val_mode=true)
view_importance(R, L, MeF, SdF)

for (fe, loc) in sort(collect(zip(MeF, get_amino_loc(R, L))), by = x -> x[1])[1:10]
    @printf "Location %s : %.4f\n" loc fe
end
```