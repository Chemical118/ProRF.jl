# Toolbox
Documentaion for `ProRF`'s useful things to perform data preprocessing or random forest.

## Random Forest
```@docs
get_rf_value
train_test_split
nrmse
test_nrmse
min_max_norm
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

|Amino acid|Molar volume \[[1](@ref Reference)\]|pI \[[2](@ref Reference)\]|Hydrophobicity \[[4](@ref Reference)\]|Frequency \[[3](@ref Reference)\]|
|:---|:---|:---|:---|:---|
|Alanine|88.6|6.00|51|0.0777|
|Arginine|173.4|10.76|-144|0.0627|
|Asparagine|114.1|5.41|-84|0.0336|
|Aspartic acid|111.1|2.77|-78|0.0542|
|Cysteine|108.5|5.07|137|0.0078|
|Glutamic acid|138.4|3.22|-115|0.0859|
|Glutamine|143.8|5.65|-128|0.0315|
|Glycine|60.1|5.97|-13|0.0730|
|Histidine|153.2|7.59|-55|0.0192|
|Isoleucine|166.7|6.02|106|0.0666|
|Leucine|166.7|5.98|103|0.0891|
|Lysine|168.6|9.74|-205|0.0776|
|Methionine|162.9|5.74|73|0.0241|
|Phenylalanine|189.9|5.48|108|0.0361|
|Proline|112.7|6.30|-79|0.0435|
|Serine|89.0|5.68|-26|0.0466|
|Threonine|116.1|5.60|-3|0.0487|
|Tryptophan|227.8|5.89|69|0.0102|
|Tyrosine|193.6|5.66|11|0.0300|
|Valine|140.0|5.96|108|0.0817|
Frequency of amino acid is used to estimate the value of unknown amino acid.
```@docs
ProRF.volume
ProRF.pI
ProRF.hydrophobicity 
```
## Others
```@docs
ProRF._julia_interactive
julia_isinteractive
ProRF.@seed
ProRF.@show_pyplot
```

## Reference
\[1\] : Brooks, D. J., Fresco, J. R., Lesk, A. M., & Singh, M. (2002). Evolution of amino acid frequencies in proteins over deep time: inferred order of introduction of amino acids into the genetic code. *Molecular Biology and Evolution, 19*(10), 1645–1655. <https://doi.org/10.1093/oxfordjournals.molbev.a003988>

\[2\] :  Lide, D. R. (Ed.). (1991). *Hdbk of chemistry & physics 72nd edition* (72nd ed.). CRC Press.

\[3\] :  Naderi-Manesh, H., Sadeghi, M., Arab, S., & Moosavi Movahedi, A. A. (2001). Prediction of protein surface accessibility with information theory. *Proteins, 42*(4), 452–459. [https://doi.org/10.1002/1097-0134(20010301)42:4<452::aid-prot40>3.0.co;2-q](https://doi.org/10.1002/1097-0134(20010301)42:4<452::aid-prot40>3.0.co;2-q)

\[4\] :  Zamyatnin, A. (1984). Amino Acid, Peptide, and Protein Volume in Solution. *Annual Review of Biophysics and Biomolecular Structure, 13*(1), 145–165. <https://doi.org/10.1146/annurev.biophys.13.1.145>