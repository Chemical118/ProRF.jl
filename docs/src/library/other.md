# Toolbox
Documentaion for `ProRF`'s useful things to perform data preprocessing or random forest.

## Random Forest
```@docs
train_test_split
nrmse
save_model
load_model
parallel_predict
```

## Data preprocessing
```@docs
view_sequence
view_mutation
```

## Others
```@docs
ProRF._julia_interactive
julia_isinteractive
ProRF.@seed_i
ProRF.@seed_u64
ProRF.@show_pyplot
```

## Convert dictionary
Convert dictionary for [`get_data`](@ref)
```@docs
ProRF.volume
ProRF.pI
ProRF.hydrophobicity 
```