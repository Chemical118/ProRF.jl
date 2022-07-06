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

## Convert dictionary
Convert dictionary for [`get_data`](@ref)

|Amino acid|Molar volume|pI|Hydrophobicity|
|:---|:---|:---|:---|
|Alanine|88.6|6.11|51|
|Arginine|173.4|10.76|-144|
|Asparagine|114.1|5.43|-84|
|Aspartic acid|111.1|2.98|-78|
|Cysteine|108.5|5.15|137|
|Glutamic acid|138.4|3.08|-115|
|Glutamine|143.8|5.65|-128|
|Glycine|60.1|6.06|-13|
|Histidine|153.2|7.64|-55|
|Isoleucine|166.7|6.04|106|
|Leucine|166.7|6.04|103|
|Lysine|168.6|9.47|-205|
|Methionine|162.9|5.71|73|
|Phenylalanine|189.9|5.76|108|
|Proline|112.7|6.30|-79|
|Serine|89.0|5.70|-26|
|Threonine|116.1|5.60|-3|
|Tryptophan|227.8|5.88|69|
|Tyrosine|193.6|5.63|11|
|Valine|140.0|6.02|108|
```@docs
ProRF.volume
ProRF.pI
ProRF.hydrophobicity 
```

## Others
```@docs
ProRF._julia_interactive
julia_isinteractive
ProRF.@seed_i
ProRF.@seed_u64
ProRF.@show_pyplot
```