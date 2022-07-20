var documenterSearchIndex = {"docs":
[{"location":"library/datapro/#Data-preprocessing","page":"Data preprocess","title":"Data preprocessing","text":"","category":"section"},{"location":"library/datapro/","page":"Data preprocess","title":"Data preprocess","text":"Documentaion for ProRF's data preprocessing function.","category":"page"},{"location":"library/datapro/","page":"Data preprocess","title":"Data preprocess","text":"data_preprocess_index\ndata_preprocess_fill","category":"page"},{"location":"library/datapro/#ProRF.data_preprocess_index","page":"Data preprocess","title":"ProRF.data_preprocess_index","text":"data_preprocess_index(in_fasta_loc::String;\n                      target_rate::Float64=0.3,\n                      val_mode::Bool=false)\n\nExamples\n\njulia> Find, Lind = data_preprocess_index(\"Data/algpdata.fasta\", val_mode=true);\n\njulia> @printf \"%d %d\\n\" Find Lind\n26 492\n\nAnalyze aligned sequence, and get index to slice aligned sequence.\n\nReturn front, last index whose gap ratio is smaller than target_rate at each ends, then display partial of both ends when val_mode is off.\n\nArguments\n\nin_fasta_loc::String : location of .fasta file\ntarget_rate::Float64 : baseline for gap ratio\nval_mode::Bool : when val_mode is true, function don't display anything.\n\n\n\n\n\n","category":"function"},{"location":"library/datapro/#ProRF.data_preprocess_fill","page":"Data preprocess","title":"ProRF.data_preprocess_fill","text":"data_preprocess_fill(front_ind::Int, last_ind::Int,\n                     in_fasta_loc::String,\n                     newick_loc::String,\n                     out_fasta_loc::String;\n                     val_mode::Bool=false)\n\nExamples\n\njulia> data_preprocess_fill(Find, Lind,\n                            \"Data/algpdata.fasta\",\n                            \"Data/Mega/ealtreedata.nwk\",\n                            \"Data/jealgpdata.fasta\",\n                            val_mode=true);\n<Closest Target --> Main>\nXP_045911656.1 --> AAR20843.1\nXP_042307646.1 --> XP_042700069.1\nXP_006741075.1 --> ABD77268.1\nAAL66372.1 --> AAH29754.1\nXP_006741075.1 --> AAA31017.1\nXP_006741075.1 --> ABD77263.1\nXP_031465919.1 --> XP_046760974.1\nXP_032600743.1 --> XP_014813383.1\nXP_033927480.1 --> PKK17119.1\n\nFill aligned partial .fasta file with nearest data without having a gap by .nwk file, then display edited data.\n\nMake sure that id of .fasta file same as id of .nwk file.\n\nArguments\n\nfront_ind::Int, last_ind::Int : front, last index by data_preprocess_index or user defined.\nin_fasta_loc::String : location of .fasta file.\nnewick_loc::String : location of .fasta file.\nout_fasta_loc::String : location of output .fasta file.\nval_mode::Bool : when val_mode is true, function don't display anything.\n\n\n\n\n\n","category":"function"},{"location":"library/other/#Toolbox","page":"Toolbox","title":"Toolbox","text":"","category":"section"},{"location":"library/other/","page":"Toolbox","title":"Toolbox","text":"Documentaion for ProRF's useful things to perform data preprocessing or random forest.","category":"page"},{"location":"library/other/#Random-Forest","page":"Toolbox","title":"Random Forest","text":"","category":"section"},{"location":"library/other/","page":"Toolbox","title":"Toolbox","text":"get_rf_value\ntrain_test_split\nnrmse\ntest_nrmse\nmin_max_norm\nsave_model\nload_model\nparallel_predict","category":"page"},{"location":"library/other/#ProRF.get_rf_value","page":"Toolbox","title":"ProRF.get_rf_value","text":"get_rf_value(X::Matrix{Float64}, Y::Vector{Float64};\n             iter::Int=10, test_size::Float64=0.3,\n             feat_range::Int=4, base_tree::Int=50,\n             memory_usage::Float64=4.0,\n             max_tree::Int=1000)\n\nExamples\n\njulia> NFeat, NTree, NDepth = get_rf_value(X, Y, iter=5, memory_usage=10);\n\nFind best three arguments for random forest.\n\nArguments\n\nX::Matrix{Float64} : X data.\nY::Vector{Float64} : Y data.\niter::Int : number of operations iterations.\ntest_size::Float64 : size of test set.\nfeat_range::Int : scope of search for number of selected features.\nbase_tree::Int : number of trees used when navigating.\nmemory_usage::Float64 : available memory capacity (GB)\nmax_tree::Int : thresholds fornumber of trees for performance.\n\nReturn\n\nopt_feat::Int : optimized number of selected features.\nopt_tree::Int : optimized number of trees.\nopt_depth::Int : optimized maximum depth of the tree.\n\n\n\n\n\n","category":"function"},{"location":"library/other/#ProRF.train_test_split","page":"Toolbox","title":"ProRF.train_test_split","text":"train_test_split(X::Matrix{Float64}, Y::Vector{Float64};\n                 test_size::Float64=0.3, \n                 data_state::UInt64=@seed)\n\nExamples\n\njulia> x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2);\n\nSplit X, Y data to train, test set.\n\nArguments\n\nX::Matrix{Float64} : X data.\nY::Vector{Float64} : Y data.\ntest_size::Float64 : size of test set.\ndata_state::UInt64 : seed used to split data.\n\n\n\n\n\n","category":"function"},{"location":"library/other/#ProRF.nrmse","page":"Toolbox","title":"ProRF.nrmse","text":"nrmse(pre::Vector{Float64}, tru::Vector{Float64})\n\nCompute normalized root mean square error with predict value and true value.\n\n\n\n\n\n","category":"function"},{"location":"library/other/#ProRF.test_nrmse","page":"Toolbox","title":"ProRF.test_nrmse","text":"test_nrmse(regr::RandomForestRegressor, X::Matrix{Float64}, Y::Vector{Float64})\n\nCompute normalized root mean square error with regression model, X and Y data.\n\n\n\n\n\ntest_nrmse(regr::RandomForestRegressor, X::Matrix{Float64}, Y::Vector{Float64},\n           data_state::UInt64;\n           test_size::Float64=0.3, test_mode::Bool=true)\n\nCompute test or train set normalized root mean square error with regression model, X, Y data and seed.\n\n\n\n\n\n","category":"function"},{"location":"library/other/#ProRF.min_max_norm","page":"Toolbox","title":"ProRF.min_max_norm","text":"min_max_norm(data::Vector{Float64})\n\nMin-max normalization function.\n\n\n\n\n\n","category":"function"},{"location":"library/other/#ProRF.save_model","page":"Toolbox","title":"ProRF.save_model","text":"save_model(model_loc::String, regr::RandomForestRegressor)\n\nExamples\n\njulia> M = rf_model(X, Y, 5, 300, val_mode=true);\n\njulia> save_model(\"model.jld\", M);\nERROR: model.jld is not .jld2 file\n\njulia> save_model(\"model.jld2\", M);\n\nSave RandomForestRegressor model using JLD2, make sure filename extension set to .jld2.\n\n\n\n\n\n","category":"function"},{"location":"library/other/#ProRF.load_model","page":"Toolbox","title":"ProRF.load_model","text":"load_model(model_loc::String)\n\nExamples\n\njulia> M = load_model(\"model.jld2\");\n\njulia> X, Y, L = get_data(R, 9, 'E');\n\njulia> @printf \"Total NRMSE : %.6f\\n\" nrmse(M, X, Y)\nTotal NRMSE : 0.136494\n\nLoad RandomForestRegressor model using JLD2, make sure filename extension set to .jld2.\n\n\n\n\n\n","category":"function"},{"location":"library/other/#ProRF.parallel_predict","page":"Toolbox","title":"ProRF.parallel_predict","text":"parallel_predict(regr::RandomForestRegressor, X::Matrix{Float64})\n\nExecute DecisionTree.predict(regr, X) in parallel.\n\n\n\n\n\nparallel_predict(regr::RandomForestRegressor, L::Vector{Int},\n                 seq_vector::Vector{String};\n                 convert::Dict{Char, Float64}=ProRF.volume)\n\nGet raw sequence vector and L data to make X data and execute DecisionTree.predict(regr, X) in parallel.\n\n\n\n\n\n","category":"function"},{"location":"library/other/#Data-preprocessing","page":"Toolbox","title":"Data preprocessing","text":"","category":"section"},{"location":"library/other/","page":"Toolbox","title":"Toolbox","text":"view_sequence\nview_mutation","category":"page"},{"location":"library/other/#ProRF.view_sequence","page":"Toolbox","title":"ProRF.view_sequence","text":"view_sequence(fasta_loc::String, amino_loc::Int=1;\n              fontsize::Int=9,\n              seq_width::Int=800,\n              save::Bool=false)\n\nDisplay .fasta file sequence using PyCall.\n\nMade it by referring to bokeh sequence aligner visualization program.\n\nArguments\n\nfasta_loc::String : location of .fasta file.\namino_loc::Int : start index for .fasta file (when value is not determined, set to 1).\nfontsize::Int : font size of sequence.\nseq_width::Int : gap width between sequences.\nsave::Bool : save .html viewer.\n\n\n\n\n\nview_sequence(R::AbstractRF;\n              fontsize::Int=9,\n              seq_width::Int=800,\n              save::Bool=false)\n\nDisplay .fasta file sequence exists at a location in the AbstractRF object using PyCall.\n\nMade it by referring to bokeh sequence aligner visualization program.\n\nArguments\n\nR::AbstractRF : for both RF and RFI.\nfontsize::Int : font size of sequence.\nseq_width::Int : gap width between sequences.\nsave::Bool : save .html viewer.\n\n\n\n\n\n","category":"function"},{"location":"library/other/#ProRF.view_mutation","page":"Toolbox","title":"ProRF.view_mutation","text":"view_mutation(fasta_loc::String)\n\nAnalyze data from .fasta file location of AbstractRF, then draw histogram, line graph about the mutation distribution.\n\n\n\n\n\nview_mutation(R::AbstractRF)\n\nAnalyze data from .fasta file, then draw histogram, line graph about the mutation distribution.\n\n\n\n\n\n","category":"function"},{"location":"library/other/#Convert-dictionary","page":"Toolbox","title":"Convert dictionary","text":"","category":"section"},{"location":"library/other/","page":"Toolbox","title":"Toolbox","text":"Convert dictionary for get_data","category":"page"},{"location":"library/other/","page":"Toolbox","title":"Toolbox","text":"Amino acid Molar volume [4] pI [2] Hydrophobicity [3] Frequency [1]\nAlanine 88.6 6.00 51 0.0777\nArginine 173.4 10.76 -144 0.0627\nAsparagine 114.1 5.41 -84 0.0336\nAspartic acid 111.1 2.77 -78 0.0542\nCysteine 108.5 5.07 137 0.0078\nGlutamic acid 138.4 3.22 -115 0.0859\nGlutamine 143.8 5.65 -128 0.0315\nGlycine 60.1 5.97 -13 0.0730\nHistidine 153.2 7.59 -55 0.0192\nIsoleucine 166.7 6.02 106 0.0666\nLeucine 166.7 5.98 103 0.0891\nLysine 168.6 9.74 -205 0.0776\nMethionine 162.9 5.74 73 0.0241\nPhenylalanine 189.9 5.48 108 0.0361\nProline 112.7 6.30 -79 0.0435\nSerine 89.0 5.68 -26 0.0466\nThreonine 116.1 5.60 -3 0.0487\nTryptophan 227.8 5.89 69 0.0102\nTyrosine 193.6 5.66 11 0.0300\nValine 140.0 5.96 108 0.0817","category":"page"},{"location":"library/other/","page":"Toolbox","title":"Toolbox","text":"Frequency of amino acid is used to estimate the value of unknown amino acid.","category":"page"},{"location":"library/other/","page":"Toolbox","title":"Toolbox","text":"ProRF.volume\nProRF.pI\nProRF.hydrophobicity ","category":"page"},{"location":"library/other/#ProRF.volume","page":"Toolbox","title":"ProRF.volume","text":"Convert dictionary about molar volume of amino acid.\n\n\n\n\n\n","category":"constant"},{"location":"library/other/#ProRF.pI","page":"Toolbox","title":"ProRF.pI","text":"Convert dictionary about pI of amino acid.\n\n\n\n\n\n","category":"constant"},{"location":"library/other/#ProRF.hydrophobicity","page":"Toolbox","title":"ProRF.hydrophobicity","text":"Convert dictionary about hydrophobicity of amino acid.\n\n\n\n\n\n","category":"constant"},{"location":"library/other/#Others","page":"Toolbox","title":"Others","text":"","category":"section"},{"location":"library/other/","page":"Toolbox","title":"Toolbox","text":"ProRF._julia_interactive\njulia_isinteractive\nProRF.@seed\nProRF.@show_pyplot","category":"page"},{"location":"library/other/#ProRF._julia_interactive","page":"Toolbox","title":"ProRF._julia_interactive","text":"Check julia currently running is interactive or not.\n\nYou can change the value in julia_isinteractive.\n\n\n\n\n\n","category":"constant"},{"location":"library/other/#ProRF.julia_isinteractive","page":"Toolbox","title":"ProRF.julia_isinteractive","text":"julia_isinteractive()\n\nCheck _julia_interactive value.\n\n\n\n\n\njulia_isinteractive(x::Bool)\n\nSet _julia_interactive value.\n\n\n\n\n\n","category":"function"},{"location":"library/other/#ProRF.@seed","page":"Toolbox","title":"ProRF.@seed","text":"Return UInt64 range integer MersenneTwister RNG object seed. keep in mind that when macro executed, the seed is initialized.\n\n\n\n\n\n","category":"macro"},{"location":"library/other/#ProRF.@show_pyplot","page":"Toolbox","title":"ProRF.@show_pyplot","text":"When _julia_interactive is on, execute display(gcf()) or _julia_interactive is off, execute show() and wait until the user inputs enter.\n\n\n\n\n\n","category":"macro"},{"location":"library/other/#Reference","page":"Toolbox","title":"Reference","text":"","category":"section"},{"location":"library/other/","page":"Toolbox","title":"Toolbox","text":"[1] : Brooks, D. J., Fresco, J. R., Lesk, A. M., & Singh, M. (2002). Evolution of amino acid frequencies in proteins over deep time: inferred order of introduction of amino acids into the genetic code. Molecular Biology and Evolution, 19(10), 1645–1655. https://doi.org/10.1093/oxfordjournals.molbev.a003988","category":"page"},{"location":"library/other/","page":"Toolbox","title":"Toolbox","text":"[2] :  Lide, D. R. (Ed.). (1991). Hdbk of chemistry & physics 72nd edition (72nd ed.). CRC Press.","category":"page"},{"location":"library/other/","page":"Toolbox","title":"Toolbox","text":"[3] :  Naderi-Manesh, H., Sadeghi, M., Arab, S., & Moosavi Movahedi, A. A. (2001). Prediction of protein surface accessibility with information theory. Proteins, 42(4), 452–459. https://doi.org/10.1002/1097-0134(20010301)42:4<452::aid-prot40>3.0.co;2-q","category":"page"},{"location":"library/other/","page":"Toolbox","title":"Toolbox","text":"[4] :  Zamyatnin, A. (1984). Amino Acid, Peptide, and Protein Volume in Solution. Annual Review of Biophysics and Biomolecular Structure, 13(1), 145–165. https://doi.org/10.1146/annurev.biophys.13.1.145","category":"page"},{"location":"library/ranfor/#Random-Forest","page":"Random Forest","title":"Random Forest","text":"","category":"section"},{"location":"library/ranfor/","page":"Random Forest","title":"Random Forest","text":"Documentaion for ProRF's functions performing a random forest.","category":"page"},{"location":"library/ranfor/#Main-Random-Forest","page":"Random Forest","title":"Main Random Forest","text":"","category":"section"},{"location":"library/ranfor/","page":"Random Forest","title":"Random Forest","text":"AbstractRF\nAbstractRFI\nRF\nRFI\nget_data\nget_reg_importance\nrf_importance\nrf_model\nrf_nrmse\niter_get_reg_importance\nview_result\nview_importance","category":"page"},{"location":"library/ranfor/#ProRF.AbstractRF","page":"Random Forest","title":"ProRF.AbstractRF","text":"Supertype for RF, RFI.\n\n\n\n\n\n","category":"type"},{"location":"library/ranfor/#ProRF.AbstractRFI","page":"Random Forest","title":"ProRF.AbstractRFI","text":"Supertype for RFI.\n\n\n\n\n\n","category":"type"},{"location":"library/ranfor/#ProRF.RF","page":"Random Forest","title":"ProRF.RF","text":"RF(dataset_loc::String)\n\nRF(fasta_loc::String, data_loc::String)\n\nRF(fasta_loc::String, data_loc::String, amino_loc::Union{Int, Vector{Int}})\n\nstruct for Main Random Forest\n\nExamples\n\njulia> R = RF(\"Data/rgpdata.fasta\", \"Data/rdata.xlsx\");\n\nFields\n\ndataset_loc::String : location of dataset, .fasta, .xlsx file name must be data. Also, you can designate amino_loc through index.txt. See example folder.\nfasta_loc::String : location of .fasta file.\ndata_loc::String : location of .xlsx file.\namino_loc::Union{Int, Vector{Int}} : start index or total index for amino acid (when value is not determined, set to 1).\n\n\n\n\n\n","category":"type"},{"location":"library/ranfor/#ProRF.RFI","page":"Random Forest","title":"ProRF.RFI","text":"RFI(R::AbstractRF,\n    nfeat::StepRange{Int, Int}, ntree::StepRange{Int, Int})\n\nRFI(dataset_loc::String,\n    nfeat::StepRange{Int, Int}, ntree::StepRange{Int, Int}))\n\nRFI(fasta_loc::String, data_loc::String,\n    nfeat::StepRange{Int, Int}, ntree::StepRange{Int, Int})\n\nRFI(fasta_loc::String, data_loc::String, amino_loc::Union{Int, Vector{Int}}\n    nfeat::StepRange{Int, Int}, ntree::StepRange{Int, Int})\n\nstruct for Random Forest Iteration.\n\nExamples\n\njulia> RI = RFI(\"Data/rgpdata.fasta\", \"Data/rdata.xlsx\", 2:1:10, 100:10:500);\n\nFields\n\ndataset_loc::String : location of dataset, .fasta, .xlsx file name must be data. Also, you can designate amino_loc through index.txt. See example folder.\nfasta_loc::String : location of .fasta file.\ndata_loc::String : location of .xlsx file.\namino_loc::Union{Int, Vector{Int}} : start index or total index for amino acid (when value is not determined, set to 1).\nnfeat::StepRange{Int, Int} : range of the number of selected features.\nntree::StepRange{Int, Int} : range of the number of trees.\n\n\n\n\n\n","category":"type"},{"location":"library/ranfor/#ProRF.get_data","page":"Random Forest","title":"ProRF.get_data","text":"get_data(R::AbstractRF, ami_arr::Int, excel_col::Char;\n         norm::Bool=false, convert::Dict{Char, Float64}=ProRF.volume,\n         sheet::String=\"Sheet1\", title::Bool=true)\n\nget_data(R::AbstractRF, excel_col::Char;\n         norm::Bool=false, convert::Dict{Char, Float64}=ProRF.volume,\n         sheet::String=\"Sheet1\", title::Bool=true)\n\nExamples\n\njulia> X, Y, L = get_data(R, 9, 'E');\n\nGet data from .fasta file by converting selected dictionary and .xlsx file at certain sheet and column.\n\nArguments\n\nR::AbstractRF : for both RF and RFI.\nami_arr::Int : baseline for total number of mutations in samples at one location (when value is not determined, set to 1).\nexcel_col::Char : column character for .xlsx file.\nnorm::Bool : execute min-max normalization.\nconvert::Dict{Char, Float64} : Convert dictionary that turns amnio acid to value\nsheet::String : .xlsx data sheet name\ntitle::Bool : when .xlsx have a header row, turn on title.\n\nReturn\n\nX::Matrix{Float64} : independent variables data matrix.\nY::Vector{Float64} : dependent variable data vector.\nL::Vector{Int} : raw sequence index vector.\n\n\n\n\n\n","category":"function"},{"location":"library/ranfor/#ProRF.get_reg_importance","page":"Random Forest","title":"ProRF.get_reg_importance","text":"get_reg_importance(R::AbstractRF, X::Matrix{Float64}, Y::Vector{Float64},\n                   L::Vector{Int}, feat::Int, tree::Int;\n                   val_mode::Bool=false, test_size::Float64=0.3,\n                   nbin::Int=200, show_number::Int=20, imp_iter::Int=60,\n                   max_depth::Int=-1,\n                   min_samples_leaf::Int=1,\n                   min_samples_split::Int=2,\n                   data_state::UInt64=@seed,\n                   learn_state::UInt64=@seed,\n                   imp_state::UInt64=@seed)\n\nExamples\n\njulia> M, F = get_reg_importance(R, X, Y, L, 6, 800);\n\njulia> @printf \"Total NRMSE : %.6f\\n\" nrmse(M, X, Y)\nTotal NRMSE : 0.136494\n\nCaculate regression model and feature importance, then draw random forest result and feature importance list.\n\nArguments\n\nR::AbstractRF : for both RF and RFI.\nX::Matrix{Float64} : X data.\nY::Vector{Float64} : Y data.\nL::Vector{Int} : L data.\nfeat::Int : number of selected features.\ntree::Int : number of trees.\nval_mode::Bool : when val_mode is true, function don't display anything.\ntest_size::Float64 : size of test set.\nnbin::Int : the number of bins for each two dimensions to execute kernel density estimation.\nshow_number::Int : number of locations to show importance.\nimp_iter::Int : number of times to repeat to caculate a feature importance.\nmax_depth::Int : maximum depth of the tree.\nmin_samples_leaf::Int : minimum number of samples required to be at a leaf node.\nmin_samples_split::Int : minimum number of samples required to split an internal node.\ndata_state::UInt64 : seed used to split data.\nlearn_state::UInt64 : seed used to caculate a regression model.\nimp_state::UInt64 : seed used to caculate a feature importance.\n\n\n\n\n\n","category":"function"},{"location":"library/ranfor/#ProRF.rf_importance","page":"Random Forest","title":"ProRF.rf_importance","text":"rf_importance(R::AbstractRF, regr::RandomForestRegressor,\n              X::Matrix{Float64}, L::Vector{Int};\n              val_mode::Bool=false,\n              show_number::Int=20, imp_iter::Int=60,\n              imp_state::UInt64=@seed)\n\nExamples\n\njulia> F = rf_importance(R, M, X, L, show_number=15);\n\nCaculate feature importance for a target model, then draw feature importance list.\n\nArguments\n\nR::AbstractRF : for both RF and RFI.\nregr::RandomForestRegressor : target regression model.\nX::Matrix{Float64} : X data.\nL::Vector{Int} : L data.\nval_mode::Bool : when val_mode is true, function don't display anything.\nshow_number::Int : number of locations to show importance.\nimp_iter::Int : number of times to repeat to caculate a feature importance.\nimp_state::UInt64 : seed used to caculate a feature importance.\n\n\n\n\n\n","category":"function"},{"location":"library/ranfor/#ProRF.rf_model","page":"Random Forest","title":"ProRF.rf_model","text":"rf_model(X::Matrix{Float64}, Y::Vector{Float64}, feat::Int, tree::Int;\n         val_mode::Bool=false, test_size::Float64=0.3, nbin::Int=200,\n         max_depth::Int=-1,\n         min_samples_leaf::Int=1,\n         min_samples_split::Int=2,\n         data_state::UInt64=@seed, \n         learn_state::UInt64=@seed)\n\nExamples\n\njulia> M = rf_model(X, Y, 6, 800);\n\nCaculate regression model, then draw random forest result.\n\nArguments\n\nX::Matrix{Float64} : X data.\nY::Vector{Float64} : Y data.\nfeat::Int : number of selected features.\ntree::Int : number of trees.\nval_mode::Bool : when val_mode is true, function don't display anything.\ntest_size::Float64 : size of test set.\nnbin::Int : the number of bins for each two dimensions to execute kernel density estimation.\nmax_depth::Int : maximum depth of the tree.\nmin_samples_leaf::Int : minimum number of samples required to be at a leaf node.\nmin_samples_split::Int : minimum number of samples required to split an internal node.\ndata_state::UInt64 : seed used to split data.\nlearn_state::UInt64 : seed used to caculate a regression model.\n\n\n\n\n\n","category":"function"},{"location":"library/ranfor/#ProRF.rf_nrmse","page":"Random Forest","title":"ProRF.rf_nrmse","text":"rf_nrmse(X::Matrix{Float64}, Y::Vector{Float64}, feat::Int, tree::Int;\n         val_mode::Bool=false, test_size::Float64=0.3, nbin::Int=200,\n         max_depth::Int=-1,\n         min_samples_leaf::Int=1,\n         min_samples_split::Int=2,\n         data_state::UInt64=@seed, \n         learn_state::UInt64=@seed)\n\nExamples\n\njulia> M, NE = rf_nrmse(X, Y, 6, 800);\n\nCaculate normalized root mean square error, then draw random forest result.\n\nArguments\n\nX::Matrix{Float64} : X data.\nY::Vector{Float64} : Y data.\nfeat::Int : number of selected features.\ntree::Int : number of trees.\nval_mode::Bool : when val_mode is true, function don't display anything.\ntest_size::Float64 : size of test set.\nnbin::Int : the number of bins for each two dimensions to execute kernel density estimation.\nmax_depth::Int : maximum depth of the tree.\nmin_samples_leaf::Int : minimum number of samples required to be at a leaf node.\nmin_samples_split::Int : minimum number of samples required to split an internal node.\ndata_state::UInt64 : seed used to split data.\nlearn_state::UInt64 : seed used to caculate a regression model.\n\n\n\n\n\n","category":"function"},{"location":"library/ranfor/#ProRF.iter_get_reg_importance","page":"Random Forest","title":"ProRF.iter_get_reg_importance","text":"iter_get_reg_importance(R::AbstractRF, X::Matrix{Float64}, Y::Vector{Float64},\n                        L::Vector{Int},\n                        feat::Int, tree::Int, iter::Int;\n                        val_mode::Bool=false, test_size::Float64=0.3,\n                        show_number::Int=20, imp_iter::Int=60,\n                        max_depth::Int=-1,\n                        min_samples_leaf::Int=1,\n                        min_samples_split::Int=2,\n                        data_state::UInt64=@seed,\n                        imp_state::UInt64=@seed,\n                        learn_state_seed::UInt64=@seed)\n\nExamples\n\njulia> MF, SF = iter_get_reg_importance(R, X, Y, L, 3, 700, 10);\n\nCalculate feature importance by repeating iter::Int times with a fixed data and importance seed, then draw feature importance list inclding standard deviation.\n\nReturns the mean and standard deviation of feature importance.\n\nArguments\n\nR::AbstractRF : for both RF and RFI.\nX::Matrix{Float64} : X data.\nY::Vector{Float64} : Y data.\nL::Vector{Int} : L data.\nfeat::Int : number of selected features.\ntree::Int : number of trees.\niter::Int : number of operations iterations.\nval_mode::Bool : when val_mode is true, function don't display anything.\ntest_size::Float64 : size of test set.\nshow_number::Int : number of locations to show importance.\nimp_iter::Int : number of times to repeat to caculate a feature importance.\nmax_depth::Int : maximum depth of the tree.\nmin_samples_leaf::Int : minimum number of samples required to be at a leaf node.\nmin_samples_split::Int : minimum number of samples required to split an internal node.\ndata_state::UInt64 : seed used to split data.\nimp_state::UInt64 : seed used to caculate a feature importance.\nlearn_state_seed::UInt64 : seed used to generate seed used to caculate a regression model.\n\n\n\n\n\n","category":"function"},{"location":"library/ranfor/#ProRF.view_result","page":"Random Forest","title":"ProRF.view_result","text":"view_result(regr::RandomForestRegressor, X::Matrix{Float64}, Y::Vector{Float64};\n            nbin::Int=200)\n\nDraw random forest result and return normalized root mean square error with X, Y data and regression model.\n\n\n\n\n\nview_result(regr::RandomForestRegressor, X::Matrix{Float64}, Y::Vector{Float64},\n            data_state::UInt64;\n            test_size::Float64=0.3, nbin::Int=200, test_mode::Bool=true)\n\nDraw test or train set random forest result and return normalized root mean square error with regression model, X, Y data and seed.\n\n\n\n\n\nview_result(pre::Vector{Float64}, tru::Vector{Float64}; nbin::Int=200)\n\nDraw random forest result and return normalized root mean square error with predict value and true value.\n\n\n\n\n\nview_result(pre::Vector{Float64}, tru::Vector{Float64}, data_state::UInt64;\n            test_size::Float64=0.3, nbin::Int=200, test_mode::Bool=true)\n\nDraw test or train set random forest result and return normalized root mean square error with predict, true value and seed.\n\n\n\n\n\n","category":"function"},{"location":"library/ranfor/#ProRF.view_importance","page":"Random Forest","title":"ProRF.view_importance","text":"view_importance(R::AbstractRF, L::Vector{Int},\n                F::Vector{Float64}; show_number::Int=20)\n\nExamples\n\njulia> view_importance(R, L, F);\n\njulia> view_importance(R, L, MF, show_number=30);\n\nDraw feature importance list.\n\nArguments\n\nR::AbstractRF : for both RF and RFI.\nL::Vector{Int} : L data.\nF::Vector{Float64} : feature importance vector.\nshow_number::Int : number of locations to show importance.\n\n\n\n\n\nview_importance(R::AbstractRF, L::Vector{Int},\n                MF::Vector{Float64}, SF::Vector{Float64};\n                show_number::Int=20)\n\nExamples\n\njulia> view_importance(R, L, MF, SF, show_number=30);\n\nDraw feature importance list with standard deviation.\n\nArguments\n\nR::AbstractRF : for both RF and RFI.\nL::Vector{Int} : L data.\nMF::Vector{Float64} : mean feature importance vector.\nSF::Vector{Float64} : standard deviation feature importance vector.\nshow_number::Int : number of locations to show importance.\n\n\n\n\n\n","category":"function"},{"location":"library/ranfor/#Random-Forest-Iteration","page":"Random Forest","title":"Random Forest Iteration","text":"","category":"section"},{"location":"library/ranfor/","page":"Random Forest","title":"Random Forest","text":"get_reg_value\nget_reg_value_loc\niter_get_reg_value\nview_reg3d","category":"page"},{"location":"library/ranfor/#ProRF.get_reg_value","page":"Random Forest","title":"ProRF.get_reg_value","text":"get_reg_value(RI::AbstractRFI, X::Matrix{Float64}, Y::Vector{Float64};\n              val_mode::Bool=false, test_size::Float64=0.3,\n              max_depth::Int=-1,\n              min_samples_leaf::Int=1,\n              min_samples_split::Int=2,\n              data_state::UInt64=@seed,\n              learn_state::UInt64=@seed)\n\nExamples\n\njulia> Z = get_reg_value(RI, X, Y, val_mode=true);\n\nCalculate nrmse value for each nfeat, ntree condition, then draw nrmse value 3D graph.\n\nArguments\n\nRI::AbstractRFI : for only RFI.\nX::Matrix{Float64} : X data.\nY::Vector{Float64} : Y data.\nval_mode::Bool : when val_mode is true, function don't display anything.\ntest_size::Float64 : size of test set.\nmax_depth::Int : maximum depth of the tree.\nmin_samples_leaf::Int : minimum number of samples required to be at a leaf node.\nmin_samples_split::Int : minimum number of samples required to split an internal node.\ndata_state::UInt64 : seed used to split data.\nlearn_state::UInt64 : seed used to caculate a regression model.\n\n\n\n\n\n","category":"function"},{"location":"library/ranfor/#ProRF.get_reg_value_loc","page":"Random Forest","title":"ProRF.get_reg_value_loc","text":"get_reg_value_loc(RI::AbstractRFI, Z::Matrix{Float64})\n\nExamples\n\njulia> @printf \"%d %d\\n\" pf.get_reg_value_loc(RI, Z)...\n7 130\n\nReturns the best arguemnts depending on the nrmse value.\n\nReturn\n\nTuple{Int, Int} : best arguemnts tuple\nInt : number of selected features.\nInt : number of trees.\n\n\n\n\n\n","category":"function"},{"location":"library/ranfor/#ProRF.iter_get_reg_value","page":"Random Forest","title":"ProRF.iter_get_reg_value","text":"iter_get_reg_value(RI::AbstractRFI, X::Matrix{Float64}, Y::Vector{Float64}, iter::Int;\n                   val_mode::Bool=false, test_size::Float64=0.3,\n                   max_depth::Int=-1,\n                   min_samples_leaf::Int=1,\n                   min_samples_split::Int=2,\n                   learn_state::UInt64=@seed,\n                   data_state_seed::UInt64=@seed)\n\nExamples\n\njulia> MZ, SZ = iter_get_reg_value(RI, X, Y, 10, val_mode=true);\n\nCalculate nrmse value for each nfeat, ntree condition by repeating iter::Int times with a fixed data seed, then draw both mean and standard deviation nrmse value 3D graph.\n\nReturns the mean and standard deviation of nrmse value.\n\nArguments\n\nRI::AbstractRFI : for only RFI.\nX::Matrix{Float64} : X data.\nY::Vector{Float64} : Y data.\nval_mode::Bool : when val_mode is true, function don't display anything.\ntest_size::Float64 : size of test set.\nmax_depth::Int : maximum depth of the tree.\nmin_samples_leaf::Int : minimum number of samples required to be at a leaf node.\nmin_samples_split::Int : minimum number of samples required to split an internal node.\nlearn_state::UInt64 : seed used to caculate a regression model.\ndata_state_seed::UInt64 : seed used to generate seed used to split data.\n\n\n\n\n\n","category":"function"},{"location":"library/ranfor/#ProRF.view_reg3d","page":"Random Forest","title":"ProRF.view_reg3d","text":"view_reg3d(RI::AbstractRFI, Z::Matrix{Float64};\n           title::Union{String, Nothing}=nothing,\n           elev::Union{Real, Nothing}=nothing,\n           azim::Union{Real, Nothing}=nothing,\n           scale::Int=2)\n\nExamples\n\njulia> view_reg3d(RI, Z, title=\"NRMSE value\", scale=3);\n\njulia> view_reg3d(RI, MZ, title=\"NRMSE value\", azim=90, scale=3);\n\njulia> view_reg3d(RI, SZ, title=\"NRMSE SD value\", elev=120, scale=3);\n\nArguments\n\nRI::AbstractRFI : for only RFI.\nZ::Matrix{Float64} : nrmse matrix.\ntitle::Union{String, Nothing} : title of the 3d graph.\nelev::Union{Real, Nothing} : elevation viewing angle.\nazim::Union{Real, Nothing} : azimuthal viewing angle.\nscale::Int : decimal place to determine the limitation value of z axis.\n\n\n\n\n\n","category":"function"},{"location":"#ProRF.jl-Documentation","page":"Introduction","title":"ProRF.jl Documentation","text":"","category":"section"},{"location":"#Overview","page":"Introduction","title":"Overview","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"ProRF provides a full process for applying the random forest model of protein sequences using DecisionTree.","category":"page"},{"location":"#Install","page":"Introduction","title":"Install","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"warning: Warning\nProRF uses Python module Bokeh, Matplotlib to provide UI. Please install these module or execute below code before add ProRF.$ pip install matplotlib\n$ pip install bokehFor more information, see PyCall main page.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"using Pkg\nPkg.add(url=\"https://github.com/Chemical118/ProRF.jl\")","category":"page"},{"location":"#Examples","page":"Introduction","title":"Examples","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"tip: Performance Tip\nProRF support parallel computing, please turn on the julia with multiple threads. This can speed up execution time fairly.$ julia --threads autoFor more information, read Multi-Threading documentaion.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"note: Note\nProRF recommends interactive mode like IJulia. If you want to run in non-interactive mode, execute below code to see graphs. However, ProRF doesn't guarantee that you can see graphs.using ProRF\njulia_isinteractive(false)","category":"page"},{"location":"#Data-preprocessing","page":"Introduction","title":"Data preprocessing","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"ProRF has a useful function for preprocessing data.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"using ProRF, Printf\n\nFind, Lind = data_preprocess_index(\"Data/algpdata.fasta\", val_mode=true)\n@printf \"%d %d\\n\" Find Lind\n\ndata_preprocess_fill(Find, Lind,\n                     \"Data/algpdata.fasta\",\n                     \"Data/Mega/ealtreedata.nwk\",\n                     \"Data/jealgpdata.fasta\",\n                     val_mode=true);\n\nview_sequence(\"Data/jealgpdata.fasta\", save=true)","category":"page"},{"location":"#Find-best-random-forest-arguments","page":"Introduction","title":"Find best random forest arguments","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"ProRF helps you find arguments for the random forest.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"using ProRF, Printf\n\nRI = RFI(\"Data/jealgpdata.fasta\", \"Data/data.xlsx\", 2:1:10, 100:10:500)\nX, Y, L = get_data(RI, 2, 'D')\n\nMeZ, SdZ = iter_get_reg_value(RI, X, Y, 10, val_mode=true)\n\nview_reg3d(RI, MeZ, title=\"NRMSE value\", azim=90, scale=3)\nview_reg3d(RI, SdZ, title=\"NRMSE SD value\", elev=120, scale=3)\n\nN_Feature, N_Tree = get_reg_value_loc(RI, MZ)\n@printf \"Best Arguments : %d %d\\n\" N_Feature N_Tree","category":"page"},{"location":"#Execute-random-forest","page":"Introduction","title":"Execute random forest","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"ProRF executes random forest flexibly and easily.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"using ProRF, Printf\n\n# Molecular mass of amino acid\nmyDict = Dict('A' => 89, 'R' => 174, 'N' => 132, 'D' => 133, 'C' => 121, 'Q' => 146,\n'E' => 147, 'G' => 75, 'H' => 155, 'I' => 131, 'L' => 131, 'K' => 146, 'M' => 149, \n'F' => 165, 'P' => 115, 'S' => 105, 'T' => 119, 'W' => 204, 'Y' => 181, 'V' => 117)\n\nR = RF(\"Data/jealgpdata.fasta\", \"Data/data.xlsx\")\nX, Y, L = get_data(R, 2, 'D', convert=myDict)\n\nM = rf_model(X, Y, N_Feature, N_Tree)\n@printf \"Total NRMSE : %.6f\\n\" nrmse(M, X, Y)\n\nMeF, SdF = iter_get_reg_importance(R, X, Y, L, N_Feature, N_Tree, 100, val_mode=true)\nview_importance(R, L, MeF, SdF)\n\nfor (fe, loc) in sort(collect(zip(MeF, get_amino_loc(R, L))), by = x -> x[1])[1:10]\n    @printf \"Location %s : %.4f\\n\" loc fe\nend","category":"page"},{"location":"library/findex/#Index","page":"Index","title":"Index","text":"","category":"section"},{"location":"library/findex/","page":"Index","title":"Index","text":"Modules = [ProRF]","category":"page"}]
}
