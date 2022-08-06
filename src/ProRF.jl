module ProRF

using ShapML, DataFrames, DecisionTree, JLD2
using PyCall, Random, Statistics, Printf, PyPlot, StatsBase
using FASTX, BioAlignments, XLSX, Phylo, AxisArrays, AverageShiftedHistograms

export AbstractRF, AbstractRFI, RF, RFI
export get_data, view_mutation, view_reg3d, view_importance, view_sequence, view_result
export train_test_split, test_nrmse, nrmse, min_max_norm, load_model, save_model, julia_isinteractive, get_rf_value, get_amino_loc
export get_reg_importance, iter_get_reg_importance, parallel_predict
export get_reg_value, get_reg_value_loc, iter_get_reg_value, rf_importance, rf_model, rf_nrmse
export data_preprocess_fill, data_preprocess_index
export @seed

"""
Check julia currently running is interactive or not.

You can change the value in [`julia_isinteractive`](@ref).
"""
_julia_interactive = true

# Struct defination

"""
Supertype for [`RF`](@ref), [`RFI`](@ref).
"""
abstract type AbstractRF end

"""
Supertype for [`RFI`](@ref).
"""
abstract type AbstractRFI <: AbstractRF end

"""
    RF(dataset_loc::String)

    RF(fasta_loc::String, data_loc::String)

    RF(fasta_loc::String, data_loc::String, amino_loc::Union{Int, Vector{Int}})

struct for Main Random Forest
# Examples
```julia-repl
julia> R = RF("Data/rgpdata.fasta", "Data/rdata.xlsx");
```

# Fields
- `dataset_loc::String` : location of dataset, `.fasta`, `.xlsx` file name must be `data`. Also, you can designate `amino_loc` through `index.txt`. See [example](https://github.com/Chemical118/ProRF_example/tree/master/Data) folder.
- `fasta_loc::String` : location of `.fasta` file.
- `data_loc::String` : location of `.xlsx` file.
- `amino_loc::Union{Int, Vector{Int}}` : start index or total index for amino acid (when value is not determined, set to 1).
"""
struct RF <: AbstractRF
    fasta_loc::String
    data_loc::String
    amino_loc::Union{Int, Vector{Int}}
end

"""
    RFI(R::AbstractRF,
        nfeat::StepRange{Int, Int}, ntree::StepRange{Int, Int})

    RFI(dataset_loc::String,
        nfeat::StepRange{Int, Int}, ntree::StepRange{Int, Int}))

    RFI(fasta_loc::String, data_loc::String,
        nfeat::StepRange{Int, Int}, ntree::StepRange{Int, Int})
    
    RFI(fasta_loc::String, data_loc::String, amino_loc::Union{Int, Vector{Int}}
        nfeat::StepRange{Int, Int}, ntree::StepRange{Int, Int})

struct for Random Forest Iteration.
# Examples
```julia-repl
julia> RI = RFI("Data/rgpdata.fasta", "Data/rdata.xlsx", 2:1:10, 100:10:500);
```

# Fields
- `dataset_loc::String` : location of dataset, `.fasta`, `.xlsx` file name must be `data`. Also, you can designate `amino_loc` through `index.txt`. See [example](https://github.com/Chemical118/ProRF_example/tree/master/Data) folder.
- `fasta_loc::String` : location of `.fasta` file.
- `data_loc::String` : location of `.xlsx` file.
- `amino_loc::Union{Int, Vector{Int}}` : start index or total index for amino acid (when value is not determined, set to 1).
- `nfeat::StepRange{Int, Int}` : range of the number of selected features.
- `ntree::StepRange{Int, Int}` : range of the number of trees.
"""
struct RFI <: AbstractRFI
    fasta_loc::String
    data_loc::String
    amino_loc::Union{Int, Vector{Int}}
    nfeat::StepRange{Int, Int}
    ntree::StepRange{Int, Int}
end

function RF(dataset_loc::String)
    l = replace(dataset_loc, "\\" => "/")
    if isfile(l * "/index.txt")
        open(l * "/index.txt") do f
            index_str = readlines(f)[1]
            if ',' ∈ index_str
                return RF(l * "/data.fasta", l * "/data.xlsx", parse.(Int, split(index_str, ',')))
            else
                return RF(l * "/data.fasta", l * "/data.xlsx", parse(Int, index_str))
            end
        end
    else
        return RF(l * "/data.fasta", l * "/data.xlsx", 1)
    end
end

function RF(fasta_loc::String, data_loc::String)
    return RF(fasta_loc, data_loc, 1)
end

function RFI(R::AbstractRF, nfeat::StepRange{Int, Int}, ntree::StepRange{Int, Int})
    return RFI(R.fasta_loc, R.data_loc, R.amino_loc, nfeat, ntree)
end

function RFI(dataset_loc::String, nfeat::StepRange{Int, Int}, ntree::StepRange{Int, Int})
    l = replace(dataset_loc, "\\" => "/")
    if isfile(l * "/index.txt")
        open(l * "/index.txt") do f
            index_str = readlines(f)[1]
            if ',' ∈ index_str
                return RFI(l * "/data.fasta", l * "/data.xlsx", parse.(Int, split(index_str, ',')), nfeat, ntree)
            else
                return RFI(l * "/data.fasta", l * "/data.xlsx", parse(Int, index_str), nfeat, ntree)
            end
        end
    else
        return RFI(l * "/data.fasta", l * "/data.xlsx", 1, nfeat, ntree)
    end
end

function RFI(fasta_loc::String, data_loc::String, nfeat::StepRange{Int, Int}, ntree::StepRange{Int, Int})
    return RFI(fasta_loc, data_loc, 1, nfeat, ntree)
end

# Macro defination

"""
Return `UInt64` range integer `MersenneTwister` RNG object seed. keep in mind that when macro executed, the seed is initialized.
"""
macro seed()
    return :(UInt64(rand(Random.seed!(), UInt64)))
end

"""
When [`_julia_interactive`](@ref) is on, execute `display(gcf())` or [`_julia_interactive`](@ref) is off, execute `show()` and wait until the user inputs enter.
"""
macro show_pyplot()
    return :(if _julia_interactive == true
                 display(gcf())
             else
                 show()
                 print("Hit <enter> to continue")
                 readline() 
             end;
             close("all"))
end

# Normal function

function _convert_dict(convert::Union{T, Vector{T}}) where T <: Dict{Char}
    if convert isa Dict{Char}
        return Vector{Dict{Char, Float64}}([convert])
    else
        return Vector{Dict{Char, Float64}}(convert)
    end
end

function _norm_dict!(dict::Dict{Char, Float64})
    norm_aa = [('A', 0.0777), ('R', 0.0627), ('N', 0.0336), ('D', 0.0542), ('C', 0.0078), ('Q', 0.0315), ('E', 0.0859), ('G', 0.0730), ('H', 0.0192), ('I', 0.0666), ('L', 0.0891), ('K', 0.0776), ('M', 0.0241), ('F', 0.0361), ('P', 0.0435), ('S', 0.0466), ('T', 0.0487), ('W', 0.0102), ('Y', 0.0300), ('V', 0.0817)]
    dict['X'] = Float64(sum(map(x -> dict[x[1]] * x[2], norm_aa)) / sum(map(x -> x[2], norm_aa)))
end

"""
    view_mutation(fasta_loc::String)

Analyze data from `.fasta` file location of AbstractRF, then draw histogram, line graph about the mutation distribution.
"""
function view_mutation(fasta_loc::String)
    _view_mutation(fasta_loc)
end

"""
    view_mutation(R::AbstractRF)

Analyze data from `.fasta` file, then draw histogram, line graph about the mutation distribution.
"""
function view_mutation(R::AbstractRF)
    _view_mutation(R.fasta_loc)
end

function _view_mutation(fasta_loc::String)
    data_len, loc_dict_vector, _ = _location_data(fasta_loc)
    loc_vector = zeros(Int, data_len)
    loc_hist_vector = Vector{Int}()
    for dict in loc_dict_vector
        max_value = maximum(values(dict))
        if max_value ≠ data_len
            loc_vector[data_len - max_value] += 1
            push!(loc_hist_vector, data_len - max_value)
        end
    end
    loc_cum_vector = reverse(cumsum(reverse(loc_vector)))
    last_ind = findfirst(isequal(0), loc_cum_vector)
    loc_vector = loc_vector[1:last_ind]
    loc_cum_vector = loc_cum_vector[1:last_ind]
    plot(collect(1:last_ind), loc_cum_vector)
    xlabel("Number of mutation")
    ylabel("Number of total amino location")
    PyPlot.title("Mutation location stacked graph")
    xticks(collect(1:last_ind), [i % 5 == 1 ? i : "" for i in 1:last_ind])
    @show_pyplot
    hist(loc_hist_vector, bins=20)
    xlim(1, last_ind - 1)
    xticks(collect(1:last_ind - 1), [i % 10 == 1 ? i : "" for i in 1:last_ind - 1])
    xlabel("Number of mutation")
    ylabel("Number of amino location")
    PyPlot.title("Mutation location histogram")
    @show_pyplot
end

"""
    nrmse(pre::Vector{Float64}, tru::Vector{Float64})

Compute normalized root mean square error with predict value and true value.
"""
function nrmse(pre::Vector{Float64}, tru::Vector{Float64})
    return L2dist(pre, tru) / ((-(-(extrema(tru)...)))) / length(tru)^0.5
end

"""
    test_nrmse(regr::RandomForestRegressor, X::Matrix{Float64}, Y::Vector{Float64})

Compute normalized root mean square error with regression model, `X` and `Y` data.
"""
function test_nrmse(regr::RandomForestRegressor, X::Matrix{Float64}, Y::Vector{Float64})
    return L2dist(parallel_predict(regr, X), Y) / ((-(-(extrema(Y)...)))) / length(Y)^0.5
end

"""
    test_nrmse(regr::RandomForestRegressor, X::Matrix{Float64}, Y::Vector{Float64},
               data_state::UInt64;
               test_size::Float64=0.3, test_mode::Bool=true)

Compute test or train set normalized root mean square error with regression model, `X`, `Y` data and seed.
"""
function test_nrmse(regr::RandomForestRegressor, X::Matrix{Float64}, Y::Vector{Float64}, data_state::UInt64; test_size::Float64=0.3, test_mode::Bool=true)
    n = length(Y)
    idx = shuffle(MersenneTwister(data_state), 1:n)
    ed_idx = test_mode ? view(idx, 1:floor(Int, test_size*n)) : view(idx, (floor(Int, test_size*n)+1):n)
    X = X[ed_idx, :]
    Y = Y[ed_idx]
    return L2dist(parallel_predict(regr, X), Y) / (-(-(extrema(Y)...))) / length(Y)^0.5
end

"""
    get_amino_loc(L::Vector{String})

Return raw sequence index.
"""
function get_amino_loc(L::Vector{String})
    return Vector{Int}(sort(collect(Set(map(x -> parse(Int, x[1:end-1]), L)))))
end

"""
    parallel_predict(regr::RandomForestRegressor, X::Matrix{Float64})

Execute `DecisionTree.predict(regr, X)` in parallel.
"""
function parallel_predict(regr::RandomForestRegressor, X::Matrix{Float64})
    return DecisionTree.apply_forest(regr.ensemble, X, use_multithreading=true)
end

"""
    parallel_predict(regr::RandomForestRegressor, L::Vector{String},
                     seq_vector::Vector{String};
                     convert::Union{Dict{Char, Float64},
                                    Vector{Dict{Char, Float64}}}=ProRF.volume)

Get raw sequence vector and `L` data to make `X` data and execute `DecisionTree.predict(regr, X)` in parallel.
"""
function parallel_predict(regr::RandomForestRegressor, L::Vector{String}, seq_vector::Vector{String}; convert::Union{T, Vector{T}}=ProRF.volume) where T <: Dict{Char}
    seq_vector = map(x -> x[get_amino_loc(L)], seq_vector)
    test_vector = [[con[i] for con in _convert_dict(convert) for i in seq] for seq in seq_vector]
    return DecisionTree.apply_forest(regr.ensemble, Matrix{Float64}(vcat(transpose.(test_vector)...)), use_multithreading=true)
end

"""
    view_sequence(fasta_loc::String, amino_loc::Int=1;
                  fontsize::Int=9,
                  seq_width::Int=800,
                  save::Bool=false)

Display `.fasta` file sequence using `PyCall`.

Made it by referring to [bokeh sequence aligner visualization program](https://dmnfarrell.github.io/bioinformatics/bokeh-sequence-aligner).

# Arguments
- `fasta_loc::String` : location of `.fasta` file.
- `amino_loc::Int` : start index for `.fasta` file (when value is not determined, set to 1).
- `fontsize::Int` : font size of sequence.
- `seq_width::Int` : gap width between sequences.
- `save::Bool` : save `.html` viewer.
"""
function view_sequence(fasta_loc::String, amino_loc::Int=1; fontsize::Int=9, seq_width::Int=800, save::Bool=false)
    seq_vector = Vector{String}()
    id_vector = Vector{String}()

    for record in open(FASTA.Reader, fasta_loc)
        push!(id_vector, FASTA.identifier(record))
        push!(seq_vector, FASTA.sequence(String, record))
    end
    _view_sequence(fasta_loc, seq_vector, id_vector, amino_loc, fontsize, seq_width, save_view=save)
end

"""
    view_sequence(R::AbstractRF;
                  fontsize::Int=9,
                  seq_width::Int=800,
                  save::Bool=false)

Display `.fasta` file sequence exists at a location in the [`AbstractRF`](@ref) object using `PyCall`.

Made it by referring to [bokeh sequence aligner visualization program](https://dmnfarrell.github.io/bioinformatics/bokeh-sequence-aligner).

# Arguments
- `R::AbstractRF` : for both [`RF`](@ref) and [`RFI`](@ref).
- `fontsize::Int` : font size of sequence.
- `seq_width::Int` : gap width between sequences.
- `save::Bool` : save `.html` viewer.
"""
function view_sequence(R::AbstractRF; fontsize::Int=9, seq_width::Int=800, save::Bool=false)
    seq_vector = Vector{String}()
    id_vector = Vector{String}()

    for record in open(FASTA.Reader, R.fasta_loc)
        push!(id_vector, FASTA.identifier(record))
        push!(seq_vector, FASTA.sequence(String, record))
    end
    _view_sequence(R.fasta_loc, seq_vector, id_vector, R.amino_loc, fontsize, seq_width, save_view=save)
end

function _view_sequence(fasta_loc::String, seq_vector::Vector{String}, id_vector::Vector{String}, amino_loc::Int=0, fontsize::Int=9, plot_width::Int=800; save_view::Bool=true)
    py"_view_sequence"(fasta_loc, seq_vector, id_vector, amino_loc - 1, fontsize=string(fontsize) * "pt", plot_width=plot_width, save_view=save_view)
end

"""
    train_test_split(X::Matrix{Float64}, Y::Vector{Float64};
                     test_size::Float64=0.3, 
                     data_state::UInt64=@seed)

# Examples
```julia-repl
julia> x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2);
```

Split `X`, `Y` data to train, test set.

# Arguments
- `X::Matrix{Float64}` : `X` data.
- `Y::Vector{Float64}` : `Y` data.
- `test_size::Float64` : size of test set.
- `data_state::UInt64` : seed used to split data.
"""
function train_test_split(X::Matrix{Float64}, Y::Vector{Float64}; test_size::Float64=0.3, data_state::UInt64=@seed)
    # https://discourse.julialang.org/t/simple-tool-for-train-test-split/473/3
    n = length(Y)
    idx = shuffle(MersenneTwister(data_state), 1:n)
    train_idx = view(idx, (floor(Int, test_size*n)+1):n)
    test_idx = view(idx, 1:floor(Int, test_size*n))
    X[train_idx,:], X[test_idx,:], Y[train_idx], Y[test_idx]
end

"""
    save_model(model_loc::String, regr::RandomForestRegressor)

# Examples
```julia-repl
julia> M = rf_model(X, Y, 5, 300, val_mode=true);

julia> save_model("model.jld", M);
ERROR: model.jld is not .jld2 file

julia> save_model("model.jld2", M);
```
Save `RandomForestRegressor` model using `JLD2`, make sure filename extension set to `.jld2`.
"""
function save_model(model_loc::String, regr::RandomForestRegressor)
    if split(model_loc, '.')[end] != "jld2"
        error(@sprintf "%s is not .jld2 file" model_loc)
    end
    @save model_loc regr
end

"""
    load_model(model_loc::String)

# Examples
```julia-repl
julia> M = load_model("model.jld2");

julia> X, Y, L = get_data(R, 9, 'E');

julia> @printf "Total NRMSE : %.6f\\n" test_nrmse(M, X, Y)
Total NRMSE : 0.136494
```
Load `RandomForestRegressor` model using `JLD2`, make sure filename extension set to `.jld2`.
"""
function load_model(model_loc::String)
    if split(model_loc, '.')[end] != "jld2"
        error(@sprintf "%s is not .jld2 file")
    end
    @load model_loc regr
    return regr
end

"""
    julia_isinteractive()

Check [`_julia_interactive`](@ref) value.
"""
function julia_isinteractive()
    return _julia_interactive
end

"""
    julia_isinteractive(x::Bool)

Set [`_julia_interactive`](@ref) value.
"""
function julia_isinteractive(x::Bool)
    global _julia_interactive = x
end

function __init__()
    py"""
    def _view_sequence(floc, seqs, ids, loc, fontsize="9pt", plot_width=800, val_mode=False, save_view=True):
        # Bokeh sequence alignment view
        # https://dmnfarrell.github.io/bioinformatics/bokeh-sequence-aligner
        # Edit by Chemical118
        from bokeh.plotting import figure, output_file, show
        from bokeh.models import ColumnDataSource, Range1d
        from bokeh.models.glyphs import Text, Rect
        from bokeh.layouts import gridplot
        from bokeh.core.properties import value

        import numpy as np

        clrs = {'E': 'red', 'D': 'red', 'P': 'orange', 'A': 'orange', 'V': 'orange', 'H': 'orange', 'M': 'orange',
                'L': 'orange', 'I': 'orange', 'G': 'orange', 'K': 'blue', 'R': 'blue', 'N': 'green', 'C': 'green',
                'T': 'green', 'Q': 'green', 'S': 'green', 'F': 'yellow', 'Y': 'yellow', 'W': 'yellow', '-': 'white',
                'X': 'white', '.': 'white'}
        
        # make sequence and id lists from the aln object
        text = [it for s in list(seqs) for it in s]
        colors = [clrs[it] for it in text]
        n = len(seqs[0])
        s = len(seqs)
        # var = .4
        x = np.arange(1 + loc, n + 1 + loc)
        y = np.arange(0, s, 1)
        # creates a 2D grid of coords from the 1D arrays
        xx, yy = np.meshgrid(x, y)
        # flattens the arrays
        gx = xx.ravel()
        gy = yy.flatten()
        # use recty for rect coords with an offset
        recty = gy + .5
        # var = 1 / s
        # now we can create the ColumnDataSource with all the arrays
        source = ColumnDataSource(dict(x=gx, y=gy, recty=recty, text=text, colors=colors))
        plot_height = len(seqs) * 15 + 50
        x_range = Range1d(loc, n + loc + 1, bounds='auto')
        if n > 99:
            viewlen = 100
        else:
            viewlen = n + 1
        # view_range is for the close up view
        view_range = (0 + loc, viewlen + loc)
        tools = "xpan, xwheel_zoom, reset, save"

        # entire sequence view (no text, with zoom)
        p = figure(title=None, plot_width=plot_width, plot_height=50,
                x_range=x_range, y_range=(0, s), tools=tools,
                min_border=0, toolbar_location='below')
        rects = Rect(x="x", y="recty", width=1, height=1, fill_color="colors",
                    line_color=None, fill_alpha=0.6)
        p.add_glyph(source, rects)
        p.yaxis.visible = False
        p.grid.visible = False

        # sequence text view with ability to scroll along x-axis
        p1 = figure(title=None, plot_width=plot_width, plot_height=plot_height,
                    x_range=view_range, y_range=ids, tools="xpan,reset",
                    min_border=0, toolbar_location='below')  # , lod_factor=1)
        glyph = Text(x="x", y="y", text="text", text_align='center', text_color="black",
                    text_font=value("arial"), text_font_size=fontsize)
        rects = Rect(x="x", y="recty", width=1, height=1, fill_color="colors",
                    line_color=None, fill_alpha=0.4)
        p1.add_glyph(source, glyph)
        p1.add_glyph(source, rects)

        p1.grid.visible = False
        p1.xaxis.major_label_text_font_style = "bold"
        p1.yaxis.minor_tick_line_width = 0
        p1.yaxis.major_tick_line_width = 0

        p = gridplot([[p], [p1]], toolbar_location='below')

        if save_view:
            output_file('.'.join(floc.split('.')[:-1]) + '.html')
        
        show(p)
    """
end

# Data preprocess

"""
    data_preprocess_index(in_fasta_loc::String;
                          target_rate::Float64=0.3,
                          val_mode::Bool=false)

# Examples
```julia-repl
julia> Find, Lind = data_preprocess_index("Data/algpdata.fasta", val_mode=true);

julia> @printf "%d %d\\n" Find Lind
26 492
```
Analyze aligned sequence, and get index to slice aligned sequence.

Return front, last index whose gap ratio is smaller than `target_rate` at each ends, then display partial of both ends when `val_mode` is off.
# Arguments
- `in_fasta_loc::String` : location of `.fasta` file
- `target_rate::Float64` : baseline for gap ratio
- `val_mode::Bool` : when `val_mode` is true, function don't display anything.
"""
function data_preprocess_index(in_fasta_loc::String; target_rate::Float64=0.3, val_mode::Bool=false)
    data_len, loc_dict_vector, _ = _location_data(in_fasta_loc)

    seq_vector = Vector{String}()
    id_vector = Vector{String}()

    for record in open(FASTA.Reader, in_fasta_loc)
        push!(id_vector, FASTA.identifier(record))
        push!(seq_vector, FASTA.sequence(String, record))
    end

    gap_fre_vector = [Float64(get(dict, '-', 0.0) / data_len) for dict in loc_dict_vector]

    front_ind = findfirst(x -> x ≤ target_rate, gap_fre_vector)
    last_ind = findlast(x -> x ≤ target_rate, gap_fre_vector)
    seq_len = last_ind - front_ind + 1
    
    seq_vector = [seq[front_ind:last_ind] for seq in seq_vector]

    if val_mode == false
        if seq_len ≥ 100
            _view_sequence(in_fasta_loc, [seq[1:45] * '.' ^ 9 * seq[end-44:end] for seq in seq_vector], id_vector, save_view=false)
        else
            _view_sequence(in_fasta_loc, seq_vector, id_vector, save_view=false)
        end
    end

    return front_ind, last_ind
end

"""
    data_preprocess_fill(front_ind::Int, last_ind::Int,
                         in_fasta_loc::String,
                         newick_loc::String,
                         out_fasta_loc::String;
                         val_mode::Bool=false)

# Examples
```julia-repl
julia> data_preprocess_fill(Find, Lind,
                            "Data/algpdata.fasta",
                            "Data/Mega/ealtreedata.nwk",
                            "Data/jealgpdata.fasta",
                            val_mode=true);
<Closest Target --> Main>
XP_045911656.1 --> AAR20843.1
XP_042307646.1 --> XP_042700069.1
XP_006741075.1 --> ABD77268.1
AAL66372.1 --> AAH29754.1
XP_006741075.1 --> AAA31017.1
XP_006741075.1 --> ABD77263.1
XP_031465919.1 --> XP_046760974.1
XP_032600743.1 --> XP_014813383.1
XP_033927480.1 --> PKK17119.1
```
Fill aligned partial `.fasta` file with nearest data without having a gap by `.nwk` file, then display edited data.

Make sure that id of `.fasta` file same as id of `.nwk` file.

# Arguments
- `front_ind::Int`, `last_ind::Int` : front, last index by [`data_preprocess_index`](@ref) or user defined.
- `in_fasta_loc::String` : location of `.fasta` file.
- `newick_loc::String` : location of `.fasta` file.
- `out_fasta_loc::String` : location of output `.fasta` file.
- `val_mode::Bool` : when `val_mode` is true, function don't display anything.
"""
function data_preprocess_fill(front_ind::Int, last_ind::Int, in_fasta_loc::String, newick_loc::String, out_fasta_loc::String; val_mode::Bool=false)
    dis_matrix = distances(open(parsenewick, newick_loc))
    seq_vector = Vector{String}()
    id_vector = Vector{String}()
    id_dict = Dict{String, Int}()

    for (ind, record) in enumerate(open(FASTA.Reader, in_fasta_loc))
        record_id = FASTA.identifier(record)
        id_dict[record_id] = ind
        push!(id_vector, record_id)
        push!(seq_vector, FASTA.sequence(String, record)[front_ind:last_ind])
    end

    if length(Set(map(length, seq_vector))) ≠ 1
        error(@sprintf "%s is not aligned, Please align your data" fasta_loc)
    end

    seq_len = length(seq_vector[1])
    if Set(id_vector) ≠ Set(AxisArrays.axes(dis_matrix, 1))
        error("Make sure the data same name of tree and id of fasta file")
    end

    gap_ind_vector = Vector{Tuple{Int, Int}}()
    isgap_vector = Vector{Bool}()
    for seq in seq_vector
        front_gap_ind = findfirst(!isequal('-'), seq)
        last_gap_ind = findlast(!isequal('-'), seq)
        push!(isgap_vector, front_gap_ind > 2 || last_gap_ind < seq_len - 1)
        push!(gap_ind_vector, (front_gap_ind, last_gap_ind))
    end

    if any(isgap_vector)
        println("<Closest Target --> Main>")
    else
        println("There are no gap in data!")
        return
    end

    dis_matrix_id_vector = AxisArrays.axes(dis_matrix, Axis{1})
    dis_matrix_isgap_vector = [isgap_vector[id_dict[id]] for id in dis_matrix_id_vector]

    edit_seq_vector = Vector{String}()
    for (ind, (front_gap_ind, last_gap_ind)) in enumerate(gap_ind_vector)
        main_seq = seq_vector[ind]
        if isgap_vector[ind]
            nogap_dis_vector = map(i -> Float64(dis_matrix_isgap_vector[i[1]] ? 1.0 : i[2]), enumerate(dis_matrix[:, id_vector[ind]]))
            min_target_ind = id_dict[dis_matrix_id_vector[argmin(nogap_dis_vector)]]
            min_target_seq = seq_vector[min_target_ind]

            if front_gap_ind > 2
                main_seq = min_target_seq[1:front_gap_ind - 1] * main_seq[front_gap_ind:end]
            end
            if last_gap_ind < seq_len - 1
                main_seq = main_seq[1:last_gap_ind] * min_target_seq[last_gap_ind + 1:end]
            end

            @printf "%s --> %s\n" id_vector[min_target_ind] id_vector[ind]
        end
        push!(edit_seq_vector, main_seq)
    end

    open(FASTA.Writer, out_fasta_loc) do io
        for (seq, id) in zip(edit_seq_vector, id_vector)
            write(io, FASTA.Record(id, seq))
        end
    end

    if val_mode == false
        _view_sequence(out_fasta_loc, edit_seq_vector, id_vector, save_view=false)
    end
end

# RF / RFI function

"""
    get_data(R::AbstractRF, ami_arr::Int, excel_col::Char; norm::Bool=false,
             convert::Union{Dict{Char, Float64},
                            Vector{Dict{Char, Float64}}}=ProRF.volume,
             sheet::String="Sheet1", title::Bool=true)

    get_data(R::AbstractRF, excel_col::Char; norm::Bool=false,
             convert::Union{Dict{Char, Float64},
                            Vector{Dict{Char, Float64}}}=ProRF.volume,
             sheet::String="Sheet1", title::Bool=true)

# Examples
```julia-repl
julia> X, Y, L = get_data(R, 9, 'E');
```

Get data from `.fasta` file by converting selected dictionary and `.xlsx` file at certain sheet and column.

# Arguments
- `R::AbstractRF` : for both [`RF`](@ref) and [`RFI`](@ref).
- `ami_arr::Int` : baseline for total number of mutations in samples at one location (when value is not determined, set to 1).
- `excel_col::Char` : column character for `.xlsx` file.
- `norm::Bool` : execute min-max normalization.
- `convert::Dict{Char, Float64}` : [Convert dictionary](@ref) that turns amnio acid to value
- `sheet::String` : `.xlsx` data sheet name
- `title::Bool` : when `.xlsx` have a header row, turn on `title`.

# Return
- `X::Matrix{Float64}` : independent variables data matrix.
- `Y::Vector{Float64}` : dependent variable data vector.
- `L::Vector{String}` : sequence index vector.
"""
function get_data(R::AbstractRF, ami_arr::Int, excel_col::Char; norm::Bool=false, convert::Union{T, Vector{T}}=ProRF.volume, sheet::String="Sheet1", title::Bool=true) where T <: Dict{Char}
    _get_data(R, ami_arr, excel_col, norm, _convert_dict(convert), sheet, title)
end

function get_data(R::AbstractRF, excel_col::Char; norm::Bool=false, convert::Union{T, Vector{T}}=ProRF.volume, sheet::String="Sheet1", title::Bool=true) where T <: Dict{Char}
    _get_data(R, 1, excel_col, norm, _convert_dict(convert), sheet, title)
end

function _get_data(R::AbstractRF, ami_arr::Int, excel_col::Char, norm::Bool, convert::Vector{Dict{Char, Float64}}, sheet::String, title::Bool)
    aa_set = Set{Char}(['M', 'P', 'K', 'Q', 'I', 'H', 'E', 'W', 'T', 'S', 'C', 'D', 'A', 'L', 'Y', 'V', 'R', 'G', 'N', 'F'])
    for con in convert
        key_set = Set{Char}(keys(con))
        if issubset(aa_set, key_set)
            if 'X' ∉ key_set
                _norm_dict!(con)
            end
        else
            error("Please check your convert dictionary")
        end
    end

    excel_data = DataFrame(XLSX.readtable(R.data_loc, sheet, infer_eltypes=title))
    excel_select_vector = excel_data[!, Int(excel_col) - Int('A') + 1]
    data_idx = findall(!ismissing, excel_select_vector)
    excel_select_vector = Vector{Float64}(excel_select_vector[data_idx])
    if norm == true
        excel_select_vector = min_max_norm(excel_select_vector)
    end

    data_len, loc_dict_vector, seq_matrix = _location_data(R.fasta_loc, data_idx)
    x_col_vector = Vector{Vector{Float64}}()
    loc_vector = Vector{Int}()
    if R.amino_loc isa Int
        for (ind, (dict, col)) in enumerate(zip(loc_dict_vector, eachcol(seq_matrix)))
            max_val = maximum(values(dict))
            if '-' ∉ keys(dict) && ami_arr ≤ data_len - max_val 
                push!(x_col_vector, [con[i] for con in convert for i in col])
                push!(loc_vector, ind + R.amino_loc - 1)
            end
        end
    else
        for (dict, col) in zip(loc_dict_vector, eachcol(seq_matrix))
            max_val = maximum(values(dict))
            if '-' ∉ keys(dict) && ami_arr ≤ data_len - max_val 
                push!(x_col_vector, [con[i] for con in convert for i in col])
            end
        end

        if length(R.amino_loc) * length(convert) == length(x_col_vector)
            loc_vector = sort(R.amino_loc)
        else
            error("Please check your location vector")
        end
    end

    sloc_vevtor = Vector{String}()
    for i in range('a', length=length(convert))
        for sloc in string.(loc_vector)
            push!(sloc_vevtor, sloc * i)
        end
    end

    x = Matrix{Float64}(hcat(x_col_vector...))
    y = Vector{Float64}(excel_select_vector)
    l = Vector{String}(sloc_vevtor)
    return x, y, l
end

function _location_data(fasta_loc::String)
    seq_vector = [collect(FASTA.sequence(String, record)) for record in open(FASTA.Reader, fasta_loc)]
    if length(Set(map(length, seq_vector))) ≠ 1
        error(@sprintf "%s is not aligned, Please align your data" fasta_loc)
    end
    seq_matrix = permutedims(hcat(seq_vector...))
    loc_vector = Vector{Dict{Char, Int}}()
    for col in eachcol(seq_matrix)
        loc_dict = Dict{Char, Int}()
        for val in col
            loc_dict[val] = get(loc_dict, val, 0) + 1
        end
        push!(loc_vector, loc_dict)
    end
    return size(seq_matrix, 1), loc_vector, seq_matrix
end

function _location_data(fasta_loc::String, data_idx::Vector{Int})
    seq_vector = [collect(FASTA.sequence(String, record)) for record in open(FASTA.Reader, fasta_loc)][data_idx]
    if length(Set(map(length, seq_vector))) ≠ 1
        error(@sprintf "%s is not aligned, Please align your data" fasta_loc)
    end
    seq_matrix = permutedims(hcat(seq_vector...))
    loc_vector = Vector{Dict{Char, Int}}()
    for col in eachcol(seq_matrix)
        loc_dict = Dict{Char, Int}()
        for val in col
            loc_dict[val] = get(loc_dict, val, 0) + 1
        end
        push!(loc_vector, loc_dict)
    end
    return size(seq_matrix, 1), loc_vector, seq_matrix
end

"""
    min_max_norm(data::Vector{Float64})

Min-max normalization function.
"""
function min_max_norm(data::Vector{Float64})
    mi, ma = extrema(data)
    return [(i - mi) / (ma - mi) for i in data]
end

"""
    get_reg_importance(R::AbstractRF, X::Matrix{Float64}, Y::Vector{Float64},
                       L::Vector{String}, feat::Int, tree::Int;
                       val_mode::Bool=false, test_size::Float64=0.3,
                       memory_usage::Real=4.0,
                       nbin::Int=200, show_number::Int=20, imp_iter::Int=60,
                       max_depth::Int=-1,
                       min_samples_leaf::Int=1,
                       min_samples_split::Int=2,
                       data_state::UInt64=@seed,
                       learn_state::UInt64=@seed,
                       imp_state::UInt64=@seed)

# Examples
```julia-repl
julia> M, F = get_reg_importance(R, X, Y, L, 6, 800);

julia> @printf "Total NRMSE : %.6f\\n" nrmse(M, X, Y)
Total NRMSE : 0.136494
```

Caculate regression model and feature importance, then draw random forest result and feature importance list.

# Arguments
- `R::AbstractRF` : for both [`RF`](@ref) and [`RFI`](@ref).
- `X::Matrix{Float64}` : `X` data.
- `Y::Vector{Float64}` : `Y` data.
- `L::Vector{String}` : `L` data.
- `feat::Int` : number of selected features.
- `tree::Int` : number of trees.
- `val_mode::Bool` : when `val_mode` is true, function don't display anything.
- `test_size::Float64` : size of test set.
- `memory_usage::Real` : available memory capacity (GB)
- `nbin::Int` : the number of bins for each two dimensions to execute kernel density estimation.
- `show_number::Int` : number of locations to show importance.
- `imp_iter::Int` : number of times to repeat to caculate a feature importance.
- `max_depth::Int` : maximum depth of the tree.
- `min_samples_leaf::Int` : minimum number of samples required to be at a leaf node.
- `min_samples_split::Int` : minimum number of samples required to split an internal node.
- `data_state::UInt64` : seed used to split data.
- `learn_state::UInt64` : seed used to caculate a regression model.
- `imp_state::UInt64` : seed used to caculate a feature importance.
"""
function get_reg_importance(R::AbstractRF, X::Matrix{Float64}, Y::Vector{Float64}, L::Vector{String}, feat::Int, tree::Int;
    val_mode::Bool=false, test_size::Float64=0.3, memory_usage::Real=4.0, nbin::Int=200, show_number::Int=20, imp_iter::Int=60,
    max_depth::Int=-1, min_samples_leaf::Int=1, min_samples_split::Int=2,
    data_state::UInt64=@seed, learn_state::UInt64=@seed, imp_state::UInt64=@seed)
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, data_state=data_state)
    regr = _randomforestregressor(feat, tree, max_depth, min_samples_leaf, min_samples_split, learn_state)
    DecisionTree.fit!(regr, x_train, y_train)

    if val_mode == false
        _view_result(regr, x_test, y_test, nbin)
    end
    memory_estimate = *(size(X)...) / 35000.0
    if memory_estimate > memory_usage
        n = size(X, 1)
        idx = shuffle(MersenneTwister(imp_state), 1:n)
        edit_idx = view(idx, 1:floor(Int, n * memory_usage / memory_estimate))
        X = X[edit_idx, :]
    end
    return regr, _rf_importance(regr, DataFrame(X, L), imp_iter, seed=imp_state, show_number=show_number, val_mode=val_mode)
end

"""
    rf_nrmse(X::Matrix{Float64}, Y::Vector{Float64}, feat::Int, tree::Int;
             val_mode::Bool=false, test_size::Float64=0.3, nbin::Int=200,
             max_depth::Int=-1,
             min_samples_leaf::Int=1,
             min_samples_split::Int=2,
             data_state::UInt64=@seed, 
             learn_state::UInt64=@seed)

# Examples
```julia-repl
julia> M, NE = rf_nrmse(X, Y, 6, 800);
```

Caculate normalized root mean square error, then draw random forest result.

# Arguments
- `X::Matrix{Float64}` : `X` data.
- `Y::Vector{Float64}` : `Y` data.
- `feat::Int` : number of selected features.
- `tree::Int` : number of trees.
- `val_mode::Bool` : when `val_mode` is true, function don't display anything.
- `test_size::Float64` : size of test set.
- `nbin::Int` : the number of bins for each two dimensions to execute kernel density estimation.
- `max_depth::Int` : maximum depth of the tree.
- `min_samples_leaf::Int` : minimum number of samples required to be at a leaf node.
- `min_samples_split::Int` : minimum number of samples required to split an internal node.
- `data_state::UInt64` : seed used to split data.
- `learn_state::UInt64` : seed used to caculate a regression model.
"""
function rf_nrmse(X::Matrix{Float64}, Y::Vector{Float64}, feat::Int, tree::Int;
    val_mode::Bool=false, test_size::Float64=0.3, nbin::Int=200,
    max_depth::Int=-1, min_samples_leaf::Int=1, min_samples_split::Int=2,
    data_state::UInt64=@seed, learn_state::UInt64=@seed)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, data_state=data_state)
    regr = _randomforestregressor(feat, tree, max_depth, min_samples_leaf, min_samples_split, learn_state)
    DecisionTree.fit!(regr, x_train, y_train)

    if val_mode == false
        nrmse_val = _view_result(regr, x_test, y_test, nbin)
    else
        nrmse_val = test_nrmse(regr, X, Y, data_state, test_size=test_size, test_mode=true)
    end
    return regr, nrmse_val
end

"""
    rf_model(X::Matrix{Float64}, Y::Vector{Float64}, feat::Int, tree::Int;
             val_mode::Bool=false, test_size::Float64=0.3, nbin::Int=200,
             max_depth::Int=-1,
             min_samples_leaf::Int=1,
             min_samples_split::Int=2,
             data_state::UInt64=@seed, 
             learn_state::UInt64=@seed)

# Examples
```julia-repl
julia> M = rf_model(X, Y, 6, 800);
```

Caculate regression model, then draw random forest result.

# Arguments
- `X::Matrix{Float64}` : `X` data.
- `Y::Vector{Float64}` : `Y` data.
- `feat::Int` : number of selected features.
- `tree::Int` : number of trees.
- `val_mode::Bool` : when `val_mode` is true, function don't display anything.
- `test_size::Float64` : size of test set.
- `nbin::Int` : the number of bins for each two dimensions to execute kernel density estimation.
- `max_depth::Int` : maximum depth of the tree.
- `min_samples_leaf::Int` : minimum number of samples required to be at a leaf node.
- `min_samples_split::Int` : minimum number of samples required to split an internal node.
- `data_state::UInt64` : seed used to split data.
- `learn_state::UInt64` : seed used to caculate a regression model.
"""
function rf_model(X::Matrix{Float64}, Y::Vector{Float64}, feat::Int, tree::Int;
    val_mode::Bool=false, test_size::Float64=0.3, nbin::Int=200,
    max_depth::Int=-1, min_samples_leaf::Int=1, min_samples_split::Int=2,
    data_state::UInt64=@seed, learn_state::UInt64=@seed)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, data_state=data_state)
    regr = _randomforestregressor(feat, tree, max_depth, min_samples_leaf, min_samples_split, learn_state)
    DecisionTree.fit!(regr, x_train, y_train)

    if val_mode == false
        _view_result(regr, x_test, y_test, nbin)
    end
    return regr
end

"""
    view_result(regr::RandomForestRegressor, X::Matrix{Float64}, Y::Vector{Float64};
                nbin::Int=200)

Draw random forest result and return normalized root mean square error with `X`, `Y` data and regression model.
"""
function view_result(regr::RandomForestRegressor, X::Matrix{Float64}, Y::Vector{Float64}; nbin::Int=200)
    _view_result(regr, X, Y, nbin)
end

"""
    view_result(regr::RandomForestRegressor, X::Matrix{Float64}, Y::Vector{Float64},
                data_state::UInt64;
                test_size::Float64=0.3, nbin::Int=200, test_mode::Bool=true)

Draw test or train set random forest result and return normalized root mean square error with regression model, `X`, `Y` data and seed.
"""
function view_result(regr::RandomForestRegressor, X::Matrix{Float64}, Y::Vector{Float64}, data_state::UInt64; test_size::Float64=0.3, nbin::Int=200, test_mode::Bool=true)
    n = length(Y)
    idx = shuffle(MersenneTwister(data_state), 1:n)
    ed_idx = test_mode ? view(idx, 1:floor(Int, test_size*n)) : view(idx, (floor(Int, test_size*n)+1):n)
    _view_result(regr, X[ed_idx, :], Y[ed_idx], nbin)
end

"""
    view_result(pre::Vector{Float64}, tru::Vector{Float64}; nbin::Int=200)

Draw random forest result and return normalized root mean square error with predict value and true value.
"""
function view_result(pre::Vector{Float64}, tru::Vector{Float64}; nbin::Int=200)
    _view_result(pre, tru, nbin)
end

"""
    view_result(pre::Vector{Float64}, tru::Vector{Float64}, data_state::UInt64;
                test_size::Float64=0.3, nbin::Int=200, test_mode::Bool=true)

Draw test or train set random forest result and return normalized root mean square error with predict, true value and seed.
"""
function view_result(pre::Vector{Float64}, tru::Vector{Float64}, data_state::UInt64; test_size::Float64=0.3, nbin::Int=200, test_mode::Bool=true)
    n = length(pre)
    idx = shuffle(MersenneTwister(data_state), 1:n)
    ed_idx = test_mode ? view(idx, 1:floor(Int, test_size*n)) : view(idx, (floor(Int, test_size*n)+1):n)
    _view_result(pre[ed_idx], tru[ed_idx], nbin)
end

function _view_result(regr::RandomForestRegressor, x_test::Matrix{Float64}, y_test::Vector{Float64}, nbin::Int)
    predict_test = parallel_predict(regr, x_test)
    nrmse_val = nrmse(predict_test, y_test)
    if length(y_test) ≤ 150
        scatter(y_test, predict_test, color="#440154", s=20)
    else
        color = Vector{Float64}()
        ke = AverageShiftedHistograms.Kernels.gaussian
        kde = ash(y_test, predict_test, nbin=nbin, kernelx=ke, kernely=ke)
        for (tru, val) in zip(y_test, predict_test)
            push!(color, AverageShiftedHistograms.pdf(kde, tru, val))
        end
        sorted_idx = sortperm(color)
        scatter(y_test[sorted_idx], predict_test[sorted_idx], c=color[sorted_idx], s=3)
        colorbar()
    end
    PyPlot.title("Random Forest Regression Result")
    xlabel("True Values")
    ylabel("Predictions")
    axis("equal")
    axis("square")
    xlim(-max(0, -xlim()[1]), xlim()[2])
    ylim(-max(0, -ylim()[1]), ylim()[2])
    annotate((@sprintf "NRMSE : %.4f" nrmse_val), (([0.97  0.03] * collect(xlim()))[1] , ([0.05 0.95] * collect(ylim()))[1]))
    plot([-1000, 1000], [-1000, 1000], color="black")
    @show_pyplot
    @printf "NRMSE : %.6f\n" nrmse_val
    return nrmse_val
end

function _view_result(predict_test::Vector{Float64}, y_test::Vector{Float64}, nbin::Int)
    nrmse_val = nrmse(predict_test, y_test)
    if length(y_test) ≤ 150
        scatter(y_test, predict_test, color="#440154", s=20)
    else
        color = Vector{Float64}()
        ke = AverageShiftedHistograms.Kernels.gaussian
        kde = ash(y_test, predict_test, nbin=nbin, kernelx=ke, kernely=ke)
        for (tru, val) in zip(y_test, predict_test)
            push!(color, AverageShiftedHistograms.pdf(kde, tru, val))
        end
        sorted_idx = sortperm(color)
        scatter(y_test[sorted_idx], predict_test[sorted_idx], c=color[sorted_idx], s=3)
        colorbar()
    end
    PyPlot.title("Random Forest Regression Result")
    xlabel("True Values")
    ylabel("Predictions")
    axis("equal")
    axis("square")
    xlim(-max(0, -xlim()[1]), xlim()[2])
    ylim(-max(0, -ylim()[1]), ylim()[2])
    annotate((@sprintf "NRMSE : %.4f" nrmse_val), (([0.97  0.03] * collect(xlim()))[1] , ([0.05 0.95] * collect(ylim()))[1]))
    plot([-1000, 1000], [-1000, 1000], color="black")
    @show_pyplot
    @printf "NRMSE : %.6f\n" nrmse_val
    return nrmse_val
end

"""
    rf_importance(R::AbstractRF, regr::RandomForestRegressor,
                  X::Matrix{Float64}, L::Vector{String};
                  val_mode::Bool=false,
                  memory_usage::Real=4.0,
                  show_number::Int=20, imp_iter::Int=60,
                  imp_state::UInt64=@seed)
    
# Examples
```julia-repl
julia> F = rf_importance(R, M, X, L, show_number=15);
```

Caculate feature importance for a target model, then draw feature importance list.

# Arguments
- `R::AbstractRF` : for both [`RF`](@ref) and [`RFI`](@ref).
- `regr::RandomForestRegressor` : target regression model.
- `X::Matrix{Float64}` : `X` data.
- `L::Vector{String}` : `L` data.
- `val_mode::Bool` : when `val_mode` is true, function don't display anything.
- `memory_usage::Real` : available memory capacity (GB)
- `show_number::Int` : number of locations to show importance.
- `imp_iter::Int` : number of times to repeat to caculate a feature importance.
- `imp_state::UInt64` : seed used to caculate a feature importance.
"""
function rf_importance(R::AbstractRF, regr::RandomForestRegressor, X::Matrix{Float64}, L::Vector{String};
    val_mode::Bool=false, memory_usage::Real=4.0, show_number::Int=20, imp_iter::Int=60, imp_state::UInt64=@seed)
    memory_estimate = *(size(X)...) / 35000.0
    if memory_estimate > memory_usage
        n = size(X, 1)
        idx = shuffle(MersenneTwister(imp_state), 1:n)
        edit_idx = view(idx, 1:floor(Int, n * memory_usage / memory_estimate))
        X = X[edit_idx, :]
    end
    return _rf_importance(regr, DataFrame(X, L), imp_iter, seed=imp_state, val_mode=val_mode, show_number=show_number)
end

function _rf_importance(regr::RandomForestRegressor, dx::DataFrame, iter::Int=60; 
                        seed::UInt64=@seed, show_number::Int=20, val_mode::Bool=false, parallel::Bool=true)
    data_shap = ShapML.shap(explain = dx,
                    model = regr,
                    predict_function = _rf_dfpredict,
                    parallel = parallel ? :features : :none,
                    sample_size = iter,
                    seed = seed)
    data_plot = combine(groupby(data_shap, :feature_name), :shap_effect => x -> mean(abs.(x)))
    baseline = data_shap.intercept[1]
    feature_importance = data_plot[!, :shap_effect_function] ./ abs(baseline)
    if val_mode == false
        _view_importance(feature_importance, data_plot[!, :feature_name], baseline, show_number=show_number)
    end
    return feature_importance
end

function _rf_dfpredict(regr::RandomForestRegressor, x::DataFrame)
    return DataFrame(y_pred = parallel_predict(regr, Matrix{Float64}(x)))
end

function _view_importance(fe::Vector{Float64}, get_loc::Vector{String}, baseline::Float64; show_number::Int=20)
    data_len = length(fe)
    show_number = min(data_len, show_number)
    sorted_idx = sortperm(fe, rev=true)
    bar_pos = collect(data_len:-1:1) .- 0.5
    barh(bar_pos[1:show_number], fe[sorted_idx][1:show_number], align="center")
    yticks(bar_pos[1:show_number], get_loc[sorted_idx][1:show_number])
    xlabel(@sprintf "|Shapley effect| (baseline = %.2f)" baseline)
    ylabel("Amino acid Location")
    PyPlot.title("Feature Importance - Mean Absolute Shapley Value")
    @show_pyplot
end

"""
    view_importance(R::AbstractRF, L::Vector{String},
                    F::Vector{Float64}; show_number::Int=20)

# Examples
```julia-repl
julia> view_importance(R, L, F);

julia> view_importance(R, L, MF, show_number=30);
```

Draw feature importance list.

# Arguments
- `R::AbstractRF` : for both [`RF`](@ref) and [`RFI`](@ref).
- `L::Vector{String}` : `L` data.
- `F::Vector{Float64}` : feature importance vector.
- `show_number::Int` : number of locations to show importance.
"""
function view_importance(R::AbstractRF, L::Vector{String}, F::Vector{Float64}; show_number::Int=20)
    _view_importance(F, L, show_number=show_number)
end

function _view_importance(fe::Vector{Float64}, get_loc::Vector{String}; show_number::Int=20)
    data_len = length(fe)
    fe ./= maximum(fe)
    show_number = min(data_len, show_number)
    sorted_idx = sortperm(fe, rev=true)
    bar_pos = collect(data_len:-1:1) .- 0.5
    barh(bar_pos[1:show_number], fe[sorted_idx][1:show_number], align="center")
    yticks(bar_pos[1:show_number], get_loc[sorted_idx][1:show_number])
    xlabel("Feature Importance")
    ylabel("Amino acid Location")
    PyPlot.title("Relative Mean Absolute Shapley Value")
    @show_pyplot
end

"""
    iter_get_reg_importance(R::AbstractRF, X::Matrix{Float64}, Y::Vector{Float64},
                            L::Vector{String},
                            feat::Int, tree::Int,
                            data_iter::Int, learn_iter::Int;
                            val_mode::Bool=false, test_size::Float64=0.3,
                            show_number::Int=20, imp_iter::Int=60,
                            max_depth::Int=-1,
                            min_samples_leaf::Int=1,
                            min_samples_split::Int=2,
                            data_state_seed::UInt64=@seed,
                            learn_state_seed::UInt64=@seed,
                            imp_state::UInt64=@seed)

# Examples
```julia-repl
julia> MF, SF = iter_get_reg_importance(R, X, Y, L, 3, 700, 10, 10);
```
Calculate feature importance by repeating `data_iter::Int` times about data seed and `learn_iter::Int` times about model seed, then draw feature importance list inclding standard deviation.

Returns the mean and standard deviation of feature importance.

# Arguments
- `R::AbstractRF` : for both [`RF`](@ref) and [`RFI`](@ref).
- `X::Matrix{Float64}` : `X` data.
- `Y::Vector{Float64}` : `Y` data.
- `L::Vector{String}` : `L` data.
- `feat::Int` : number of selected features.
- `tree::Int` : number of trees.
- `data_iter::Int` : number of operations iterations for seed used to split data.
- `learn_iter::Int` : number of operations iterations for seed used to caculate a regression model.
- `val_mode::Bool` : when `val_mode` is true, function don't display anything.
- `test_size::Float64` : size of test set.
- `memory_usage::Real` : available memory capacity (GB)
- `show_number::Int` : number of locations to show importance.
- `imp_iter::Int` : number of times to repeat to caculate a feature importance.
- `max_depth::Int` : maximum depth of the tree.
- `min_samples_leaf::Int` : minimum number of samples required to be at a leaf node.
- `min_samples_split::Int` : minimum number of samples required to split an internal node.
- `data_state_seed::UInt64` : seed used to generate seed used to split data.
- `learn_state_seed::UInt64` : seed used to generate seed used to caculate a regression model.
- `imp_state::UInt64` : seed used to caculate a feature importance.
"""
function iter_get_reg_importance(R::AbstractRF, X::Matrix{Float64}, Y::Vector{Float64}, L::Vector{String}, feat::Int, tree::Int, data_iter::Int, learn_iter::Int;
    val_mode::Bool=false, test_size::Float64=0.3, memory_usage::Real=4.0, show_number::Int=20, imp_iter::Int=60,
    max_depth::Int=-1, min_samples_leaf::Int=1, min_samples_split::Int=2,
    data_state_seed::UInt64=@seed, learn_state_seed::UInt64=@seed, imp_state::UInt64=@seed)

    iter = learn_iter * data_iter
    f = Array{Float64}(undef, (length(L), iter))
    n = Array{Float64}(undef, iter)

    data_state_vector = Vector{UInt64}(rand(MersenneTwister(data_state_seed), UInt64, data_iter))
    learn_state_vector = Vector{UInt64}(rand(MersenneTwister(learn_state_seed), UInt64, learn_iter))
    
    memory_estimate = *(size(X)...) / 35000.0
    data_len = size(X, 1)
    if memory_estimate > memory_usage
        idx = shuffle(MersenneTwister(imp_state), 1:data_len)
        edit_idx = Vector{Int}(idx[1:floor(Int, data_len * memory_usage / memory_estimate)])
    else
        edit_idx = Vector{Int}(1:data_len)
    end

    for (i, data_state) in enumerate(data_state_vector)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, data_state=data_state)
        for (j, learn_state) in enumerate(learn_state_vector)
            idx = (i - 1) * j + j
            f[:, idx], n[idx] = _iter_get_reg_importance(X, x_train, x_test, y_train, y_test, L, feat, tree, imp_iter, max_depth, min_samples_leaf, min_samples_split, imp_state, learn_state, edit_idx)
        end
    end

    mf = mean(f, dims=2)[:, 1]
    if iter > 1
        sf = std(f, dims=2)[:, 1]
    else
        sf = zeros(Float64, size(mf))
    end

    if val_mode == false
        _iter_view_importance(mf, sf, L, show_number=show_number)
        @printf "NRMSE : %.6f\n" mean(n)
    end

    return mf, sf
end

function _iter_get_reg_importance(X::Matrix{Float64}, x_train::Matrix{Float64}, x_test::Matrix{Float64}, y_train::Vector{Float64}, y_test::Vector{Float64}, loc::Vector{String}, feat::Int, tree::Int, imp_iter::Int, max_depth::Int, min_samples_leaf::Int, min_samples_split::Int, imp_state::UInt64, learn_state::UInt64, edit_idx::Vector{Int})
    regr = _randomforestregressor(feat, tree, max_depth, min_samples_leaf, min_samples_split, learn_state)
    DecisionTree.fit!(regr, x_train, y_train)
    return _rf_importance(regr, DataFrame(X[edit_idx, :], loc), imp_iter, seed=imp_state, val_mode=true, parallel=false), test_nrmse(regr, x_test, y_test)
end

"""
    view_importance(R::AbstractRF, L::Vector{String},
                    MF::Vector{Float64}, SF::Vector{Float64};
                    show_number::Int=20)

# Examples
```julia-repl
julia> view_importance(R, L, MF, SF, show_number=30);
```

Draw feature importance list with standard deviation.

# Arguments
- `R::AbstractRF` : for both [`RF`](@ref) and [`RFI`](@ref).
- `L::Vector{String}` : `L` data.
- `MF::Vector{Float64}` : mean feature importance vector.
- `SF::Vector{Float64}` : standard deviation feature importance vector.
- `show_number::Int` : number of locations to show importance.
"""
function view_importance(R::AbstractRF, L::Vector{String}, MF::Vector{Float64}, SF::Vector{Float64}; show_number::Int=20)
    _iter_view_importance(MF, SF, L, show_number=show_number)
end

function _iter_view_importance(fe::Vector{Float64}, err::Vector{Float64}, loc::Vector{String}; show_number::Int=20)
    data_len = length(fe)
    norm_val = maximum(fe)
    fe ./= norm_val
    err ./= norm_val
    show_number = min(data_len, show_number)
    sorted_idx = sortperm(fe, rev=true)
    bar_pos = collect(data_len:-1:1) .- 0.5
    barh(bar_pos[1:show_number], fe[sorted_idx][1:show_number], xerr=err[sorted_idx][1:show_number], align="center", capsize=2)
    yticks(bar_pos[1:show_number], loc[sorted_idx][1:show_number])
    xlabel("Feature Importance")
    ylabel("Amino acid Location")
    PyPlot.title("Relative Mean Absolute Shapley Value")
    @show_pyplot
end

# RFI function

"""
    get_reg_value(RI::AbstractRFI, X::Matrix{Float64}, Y::Vector{Float64};
                  val_mode::Bool=false, test_size::Float64=0.3,
                  max_depth::Int=-1,
                  min_samples_leaf::Int=1,
                  min_samples_split::Int=2,
                  data_state::UInt64=@seed,
                  learn_state::UInt64=@seed)

# Examples
```julia-repl
julia> Z = get_reg_value(RI, X, Y, val_mode=true);
```  
Calculate [`nrmse`](@ref) value for each `nfeat`, `ntree` condition, then draw [`nrmse`](@ref) value 3D graph.

# Arguments
- `RI::AbstractRFI` : for only [`RFI`](@ref).
- `X::Matrix{Float64}` : `X` data.
- `Y::Vector{Float64}` : `Y` data.
- `val_mode::Bool` : when `val_mode` is true, function don't display anything.
- `test_size::Float64` : size of test set.
- `max_depth::Int` : maximum depth of the tree.
- `min_samples_leaf::Int` : minimum number of samples required to be at a leaf node.
- `min_samples_split::Int` : minimum number of samples required to split an internal node.
- `data_state::UInt64` : seed used to split data.
- `learn_state::UInt64` : seed used to caculate a regression model.
"""
function get_reg_value(RI::AbstractRFI, X::Matrix{Float64}, Y::Vector{Float64};
    val_mode::Bool=false, test_size::Float64=0.3,
    max_depth::Int=-1, min_samples_leaf::Int=1, min_samples_split::Int=2,
    data_state::UInt64=@seed, learn_state::UInt64=@seed)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, data_state=data_state)
    z = Array{Float64}(undef, (length(RI.nfeat), length(RI.ntree)))
    task = [(i[1], j[1], i[2], j[2]) for i in enumerate(RI.nfeat), j in enumerate(RI.ntree)]
    
    for (i, j, feat, tree) in task
        z[i,  j] = _get_reg_value(x_train, x_test, y_train, y_test, feat, tree, max_depth, min_samples_leaf, min_samples_split, learn_state)
    end

    if val_mode == false
        view_reg3d(RI, z, title="NRMSE value")
    end
    return z
end

function _get_reg_value(x_train::Matrix{Float64}, x_test::Matrix{Float64}, y_train::Vector{Float64}, y_test::Vector{Float64}, feat::Int, tree::Int, max_depth::Int, min_samples_leaf::Int, min_samples_split::Int, learn_state::UInt64)
    regr = _randomforestregressor(feat, tree, max_depth, min_samples_leaf, min_samples_split, learn_state)
    DecisionTree.fit!(regr, x_train, y_train)
    return test_nrmse(regr, x_test, y_test)
end

"""
    get_reg_value_loc(RI::AbstractRFI, Z::Matrix{Float64})

# Examples
```julia-repl
julia> @printf "%d %d\\n" pf.get_reg_value_loc(RI, Z)...
7 130
```
Returns the best arguemnts depending on the [`nrmse`](@ref) value.

# Return
- `Tuple{Int, Int}` : best arguemnts tuple
    + `Int` : number of selected features.
    + `Int` : number of trees.
"""
function get_reg_value_loc(RI::AbstractRFI, Z::Matrix{Float64})
    i, j = Tuple(findmin(Z)[2])
    return collect(RI.nfeat)[i], collect(RI.ntree)[j] 
end

"""
    view_reg3d(RI::AbstractRFI, Z::Matrix{Float64};
               title::Union{String, Nothing}=nothing,
               elev::Union{Real, Nothing}=nothing,
               azim::Union{Real, Nothing}=nothing,
               scale::Int=2)

# Examples
```julia-repl
julia> view_reg3d(RI, Z, title="NRMSE value", scale=3);

julia> view_reg3d(RI, MZ, title="NRMSE value", azim=90, scale=3);

julia> view_reg3d(RI, SZ, title="NRMSE SD value", elev=120, scale=3);
```

# Arguments
- `RI::AbstractRFI` : for only [`RFI`](@ref).
- `Z::Matrix{Float64}` : [`nrmse`](@ref) matrix.
- `title::Union{String, Nothing}` : title of the 3d graph.
- `elev::Union{Real, Nothing}` : elevation viewing angle.
- `azim::Union{Real, Nothing}` : azimuthal viewing angle.
- `scale::Int` : decimal place to determine the limitation value of z axis.
"""
function view_reg3d(RI::AbstractRFI, Z::Matrix{Float64}; title::Union{String, Nothing}=nothing, elev::Union{Real, Nothing}=nothing, azim::Union{Real, Nothing}=nothing, scale::Int=2)
    nfeat_list = collect(RI.nfeat)' .* ones(length(RI.ntree))
    ntree_list = ones(length(RI.nfeat))' .* collect(RI.ntree)
    fig = figure()
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=elev, azim=azim)
    xlabel("Numer of subfeatures")
    ylabel("Number of trees")
    mp = 10^scale
    ax.set_zlim(floor(minimum(Z)*mp)/mp, ceil(maximum(Z)*mp)/mp)
    ax.zaxis.set_major_locator(matplotlib.ticker.LinearLocator(6))
    surf = plot_surface(nfeat_list, ntree_list, Z', cmap=ColorMap("coolwarm"), linewidth=0)
    colorbar(surf, shrink=0.7, aspect=15, pad=0.1, ax=ax)
    PyPlot.title(title)
    @show_pyplot
end

"""
    iter_get_reg_value(RI::AbstractRFI, X::Matrix{Float64}, Y::Vector{Float64}, iter::Int;
                       val_mode::Bool=false, test_size::Float64=0.3,
                       max_depth::Int=-1,
                       min_samples_leaf::Int=1,
                       min_samples_split::Int=2,
                       learn_state::UInt64=@seed,
                       data_state_seed::UInt64=@seed)

# Examples
```julia-repl
julia> MZ, SZ = iter_get_reg_value(RI, X, Y, 10, val_mode=true);
```  
Calculate [`nrmse`](@ref) value for each `nfeat`, `ntree` condition by repeating `iter::Int` times with a fixed data seed, then draw both mean and standard deviation [`nrmse`](@ref) value 3D graph.

Returns the mean and standard deviation of [`nrmse`](@ref) value.

# Arguments
- `RI::AbstractRFI` : for only [`RFI`](@ref).
- `X::Matrix{Float64}` : `X` data.
- `Y::Vector{Float64}` : `Y` data.
- `val_mode::Bool` : when `val_mode` is true, function don't display anything.
- `test_size::Float64` : size of test set.
- `max_depth::Int` : maximum depth of the tree.
- `min_samples_leaf::Int` : minimum number of samples required to be at a leaf node.
- `min_samples_split::Int` : minimum number of samples required to split an internal node.
- `learn_state::UInt64` : seed used to caculate a regression model.
- `data_state_seed::UInt64` : seed used to generate seed used to split data.
"""
function iter_get_reg_value(RI::AbstractRFI, X::Matrix{Float64}, Y::Vector{Float64}, iter::Int;
    max_depth::Int=-1, min_samples_leaf::Int=1, min_samples_split::Int=2,
    val_mode::Bool=false, test_size::Float64=0.3, learn_state::UInt64=@seed, data_state_seed::UInt64=@seed)
    z = Array{Float64}(undef, (length(RI.nfeat), length(RI.ntree), iter))
    data_state_vector = Vector{UInt64}(rand(MersenneTwister(data_state_seed), UInt64, iter))
    for i = 1:iter
        z[:, :, i] = get_reg_value(RI, X, Y, val_mode=true, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, test_size=test_size, data_state=data_state_vector[i], learn_state=learn_state)
    end
    
    vz = mean(z, dims=3)[:, :, 1]
    if iter > 1
        sz = std(z, dims=3)[:, :, 1]
    else
        sz = zeros(Float64, size(vz))
    end

    if val_mode == false
        view_reg3d(RI, vz, title="NRMSE value", scale=2)
        if iter > 1
            view_reg3d(RI, sz, title="NRMSE SD value", scale=3)
        end
    end

    return vz, sz
end

function _randomforestregressor(feat::Int, tree::Int, max_depth::Int, min_samples_leaf::Int, min_samples_split::Int, learn_state::UInt64)
    return RandomForestRegressor(n_trees=tree, n_subfeatures=feat, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, rng=MersenneTwister(learn_state), impurity_importance=false)
end

"""
    get_rf_value(X::Matrix{Float64}, Y::Vector{Float64};
                 val_mode::Bool=false,
                 iter::Int=10, test_size::Float64=0.3,
                 feat_range::Int=4, base_tree::Int=50,
                 memory_usage::Real=4.0,
                 max_tree::Int=1000)

# Examples
```julia-repl
julia> NFeat, NTree, NDepth = get_rf_value(X, Y, iter=5, memory_usage=10, val_mode=true);
```
Find best three arguments for random forest.

# Arguments
- `X::Matrix{Float64}` : `X` data.
- `Y::Vector{Float64}` : `Y` data.
- `val_mode::Bool` : when `val_mode` is true, function don't display anything.
- `iter::Int` : number of operations iterations.
- `test_size::Float64` : size of test set.
- `feat_range::Int` : scope of search for number of selected features.
- `base_tree::Int` : number of trees used when navigating.
- `memory_usage::Real` : available memory capacity (GB)
- `max_tree::Int` : thresholds fornumber of trees for performance.

# Return
- `opt_feat::Int` : optimized number of selected features.
- `opt_tree::Int` : optimized number of trees.
- `opt_depth::Int` : optimized maximum depth of the tree.
"""
function get_rf_value(X::Matrix{Float64}, Y::Vector{Float64};
    val_mode::Bool=false, iter::Int=10, test_size::Float64=0.3, feat_range::Int=4, base_tree::Int=50, memory_usage::Real=4.0, max_tree::Int=1000)
    nfea = size(X, 2)
    sfea = floor(Int, sqrt(nfea))
    rfea = max(2, sfea - feat_range):min(nfea, sfea + feat_range)

    ans = Vector{Vector{Float64}}()
    for i in 1:iter
        if val_mode == false
            @printf "Find the number of feature : %d / %d\n" i iter
        end
        data_state, learn_state = @seed, @seed
        nrmse_vector = Vector{Float64}()
        for fea in rfea
            push!(nrmse_vector, rf_nrmse(X, Y, fea, base_tree, val_mode=true, data_state=data_state, learn_state=learn_state, test_size=test_size)[2])
        end
        push!(ans, nrmse_vector)
    end
    opt_feat = rfea[argmin(mean.(eachrow(Matrix{Float64}(hcat(ans...)))))]

    adep = mean([DecisionTree.depth(tree) for tree in rf_model(X, Y, opt_feat, base_tree, val_mode=true, test_size=test_size).ensemble.trees])
    idep = ceil(Int, adep / 50)
    rdep = 50:50:idep*50

    ans = Vector{Vector{Float64}}()
    for i in 1:iter
        if val_mode == false
            @printf "Find the maximum depth of the tree approximatively : %d / %d\n" i iter
        end
        data_state, learn_state = @seed, @seed
        nrmse_vector = Vector{Float64}()
        for dep in rdep
            push!(nrmse_vector, rf_nrmse(X, Y, opt_feat, base_tree, max_depth=dep, val_mode=true, data_state=data_state, learn_state=learn_state, test_size=test_size)[2])
        end
        push!(ans, nrmse_vector)
    end
    dep_ind = argmin(mean.(eachrow(Matrix{Float64}(hcat(ans...)))))

    rdep_st = rdep[max(dep_ind - 1, 1)]
    rdep_nd = min(rdep[min(dep_ind + 1, idep)], ceil(Int, adep / 10) * 10)
    if rdep_st < rdep_nd
        rdep = rdep_st:10:rdep_nd
        ans = Vector{Vector{Float64}}()
        for i in 1:iter
            if val_mode == false
                @printf "Find the maximum depth of the tree accurately : %d / %d\n" i iter
            end
            data_state, learn_state = @seed, @seed
            nrmse_vector = Vector{Float64}()
            for dep in rdep
                push!(nrmse_vector, rf_nrmse(X, Y, opt_feat, base_tree, max_depth=dep, val_mode=true, data_state=data_state, learn_state=learn_state, test_size=test_size)[2])
            end
            push!(ans, nrmse_vector)
        end
        opt_depth = rdep[argmin(mean.(eachrow(Matrix{Float64}(hcat(ans...)))))]
    else
        opt_depth = -1
    end

    atree = mean([DecisionTree.length(tree) for tree in rf_model(X, Y, opt_feat, base_tree, max_depth=opt_depth, val_mode=true, test_size=test_size).ensemble.trees])
    opt_tree = min(max_tree, floor(Int, 600000.0 * memory_usage / atree))
    return opt_feat, opt_tree, opt_depth
end

# Convert dictionary

"""
Convert dictionary about molar volume of amino acid.
"""
volume = Dict{Char, Float64}('A' => 88.6, 'R' => 173.4, 'N' => 114.1, 'D' => 111.1, 'C' => 108.5, 'Q' => 143.8, 'E' => 138.4, 'G' => 60.1, 'H' => 153.2, 'I' => 166.7, 'L' => 166.7, 'K' => 168.6, 'M' => 162.9, 'F' => 189.9, 'P' => 112.7, 'S' => 89, 'T' => 116.1, 'W' => 227.8, 'Y' => 193.6, 'V' => 140)
_norm_dict!(volume)

"""
Convert dictionary about pI of amino acid.
"""
pI = Dict{Char, Float64}('A' => 6.11, 'R' => 10.76, 'N' => 5.43, 'D' => 2.98, 'C' => 5.15, 'E' => 3.08, 'Q' => 5.65, 'G' => 6.06, 'H' => 7.64, 'I' => 6.04, 'L' => 6.04, 'K' => 9.47, 'M' => 5.71, 'F' => 5.76, 'P' => 6.30, 'S' => 5.70, 'T' => 5.60, 'W' => 5.88, 'Y' => 5.63, 'V' => 6.02)
_norm_dict!(pI)

"""
Convert dictionary about hydrophobicity of amino acid.
"""
hydrophobicity = Dict{Char, Float64}('C' => 137, 'I' => 106, 'V' => 108, 'L' => 103, 'F' => 108, 'M' => 73, 'W' => 69, 'A' => 51, 'T' => -3, 'G' => -13, 'Y' => 11, 'S' => -26, 'H' => -55, 'P' => -79, 'N' => -84, 'D' => -78, 'E' => -115, 'Q' => -128, 'R' => -144, 'K' => -205)
_norm_dict!(hydrophobicity)

end # module end