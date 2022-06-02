module ProRF

using ShapML, DataFrames, DecisionTree, JLD2
using PyCall, Random, Statistics, Printf, PyPlot, StatsBase
using FASTX, BioAlignments, XLSX, Phylo, AxisArrays

export AbstractRF, AbstractRFI, RF, RFI
export blosum_matrix, get_data, get_amino_loc, view_mutation, view_reg3d, view_importance, view_sequence
export train_test_split, nrmse, install_python_dependency, load_model, save_model, julia_isinteractive
export get_reg_importance, iter_get_reg_importance, parallel_predict
export get_reg_value, get_reg_value_loc, iter_get_reg_value, rf_importance, rf_model
export data_preprocess_fill, data_preprocess_index

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
    RF(fasta_loc::String, data_loc::String)
    RF(fasta_loc::String, data_loc::String, amino_loc::Int)

struct for Main Random Forest
# Examples
```julia-repl
julia> R = RF("Data/rgpdata.fasta", "Data/rdata.xlsx");
```

# Fields
- `fasta_loc::String` : location of `.fasta` file.
- `data_loc::String` : location of `.xlsx` file.
- `amino_loc::Int` : start index for substring (when value is not determined, set to 1).
"""
struct RF <: AbstractRF
    fasta_loc::String
    data_loc::String
    amino_loc::Int
end

"""
    RFI(fasta_loc::String, data_loc::String,
        nfeat::StepRange{Int64, Int64}, ntree::StepRange{Int64, Int64})
    
    RFI(fasta_loc::String, data_loc::String, amino_loc::Int,
        nfeat::StepRange{Int64, Int64}, ntree::StepRange{Int64, Int64})

struct for Random Forest Iteration.
# Examples
```julia-repl
julia> RI = RFI("Data/rgpdata.fasta", "Data/rdata.xlsx", 2:1:10, 100:10:500);
```

# Fields
- `fasta_loc::String` : location of `.fasta` file.
- `data_loc::String` : location of `.xlsx` file.
- `amino_loc::Int` : start index for substring (when value is not determined, set to 1).
- `nfeat::StepRange{Int64, Int64}` : range of the number of selected features.
- `ntree::StepRange{Int64, Int64}` : range of the number of trees.
"""
struct RFI <: AbstractRFI
    fasta_loc::String
    data_loc::String
    amino_loc::Int
    nfeat::StepRange{Int64, Int64}
    ntree::StepRange{Int64, Int64}
end

function RF(fasta_loc::String, data_loc::String)
    return RF(fasta_loc, data_loc, 1)
end

function RFI(fasta_loc::String, data_loc::String, nfeat::StepRange{Int64, Int64}, ntree::StepRange{Int64, Int64})
    return RFI(fasta_loc, data_loc, 1, nfeat, ntree)
end

# Macro defination

"""
Return non-negative `Int` range integer `MersenneTwister` RNG object seed. keep in mind that when macro executed, the seed is initialized.
"""
macro seed_i()
    return :(Int(rand(Random.seed!(), 0:typemax(Int))))
end

"""
Return `UInt64` range integer `MersenneTwister` RNG object seed. keep in mind that when macro executed, the seed is initialized.
"""
macro seed_u64()
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

"""
    blosum_matrix(num::Int)

# Examples
```julia-repl
julia> blosum_matrix(90)['A', 'A']
5

julia> blosum_matrix(82)
ERROR: There are no Matrix such as BLOSUM82
```
Return correct `BioAlignments` BLOSUM matrix.

Possible matrices are BLOSUM45, BLOSUM50, BLOSUM62, BLOSUM80, BLOSUM90.
"""
function blosum_matrix(num::Int)
    if num == 45
        return BLOSUM45
    elseif num == 50
        return BLOSUM50
    elseif num == 62
        return BLOSUM62
    elseif num == 80
        return BLOSUM80
    elseif num == 90
        return BLOSUM90
    else
        error(@sprintf "There are no Matrix such as BLOSUM%d" num)
    end
end

"""
    view_mutation(fasta_loc::String)

Analyze data from .fasta file location of AbstractRF, then draw histogram, line graph about the mutation distribution.
"""
function view_mutation(fasta_loc::String)
    _view_mutation(fasta_loc)
end

"""
    view_mutation(R::AbstractRF)

Analyze data from .fasta file, then draw histogram, line graph about the mutation distribution.
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
    loc_cum_vector = cumsum(loc_vector[end:-1:1])[end:-1:1]
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
    return L2dist(pre, tru) / (maximum(tru) - minimum(tru)) / length(tru)^0.5
end

"""
    nrmse(regr::RandomForestRegressor, X::Matrix{Float64}, tru::Vector{Float64})

Compute normalized root mean square error with regression model, `X` data and true value.
"""
function nrmse(regr::RandomForestRegressor, X::Matrix{Float64}, tru::Vector{Float64})
    return L2dist(parallel_predict(regr, X), tru) / (maximum(tru) - minimum(tru)) / length(tru)^0.5
end

"""
    parallel_predict(regr::RandomForestRegressor, X::Matrix{Float64})

Execute `DecisionTree.predict(regr, X)` in parallel.
"""
function parallel_predict(regr::RandomForestRegressor, X::Matrix{Float64})
    test_vector = [Vector{Float64}(row) for row in eachrow(X)]
    val_vector = similar(test_vector, Float64)

    Threads.@threads for (ind, vec) in collect(enumerate(test_vector))
        val_vector[ind] = DecisionTree.predict(regr, vec)
    end
    return val_vector
end

"""
    parallel_predict(regr::RandomForestRegressor, X::Matrix{Float64})

Get raw sequence vector and `L` data to make `X` data and execute `DecisionTree.predict(regr, X)` in parallel.
"""
function parallel_predict(regr::RandomForestRegressor, L::Vector{Tuple{Int, Char}}, seq_vector::Vector{String}; blosum::Int=62)
    seq_vector = map(x -> x[map(y -> y[1], L)], seq_vector)

    blo = blosum_matrix(blosum)
    test_vector = [[Float64(blo[tar, s]) for ((_, tar), s) in zip(L, seq)] for seq in seq_vector]
    val_vector = similar(test_vector, Float64)
    
    Threads.@threads for (ind, vec) in collect(enumerate(test_vector))
        val_vector[ind] = DecisionTree.predict(regr, vec)
    end
    return val_vector
end

"""
    view_sequence(fasta_loc::String, amino_loc::Int=1;
                  fontsize::Int=9,
                  seq_width::Int=800,
                  save::Bool=false)

Display `.fasta` file sequence by using `PyCall`.

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

Display `.fasta` file sequence exists at a location in the [`AbstractRF`](@ref) object by using `PyCall`.

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
                     data_state::UInt64=@seed_u64)

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
function train_test_split(X::Matrix{Float64}, Y::Vector{Float64}; test_size::Float64=0.3, data_state::UInt64=@seed_u64)
    # https://discourse.julialang.org/t/simple-tool-for-train-test-split/473/3
    n = length(Y)
    idx = shuffle(MersenneTwister(data_state), 1:n)
    train_idx = view(idx, (floor(Int, test_size*n)+1):n)
    test_idx = view(idx, 1:floor(Int, test_size*n))
    X[train_idx,:], X[test_idx,:], Y[train_idx], Y[test_idx]
end

"""
Install python dependency module `numpy`, `matplotlib`, `bokeh`.
"""
function install_python_dependency()
    py"_python_install"("numpy")
    py"_python_install"("matplotlib")
    py"_python_install"("bokeh")
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
Save `RandomForestRegressor` model by using `JLD2`, make sure filename extension set to `.jld2`.
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

julia> @printf "Total NRMSE : %.6f\\n" nrmse(M, X, Y)
Total NRMSE : 0.136494
```
Load `RandomForestRegressor` model by using `JLD2`, make sure filename extension set to `.jld2`.
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
    julia_isinteractive(x::Bool)

Check [`_julia_interactive`](@ref) value or set [`_julia_interactive`](@ref) value when the `Bool` input given.
"""
function julia_isinteractive()
    return _julia_interactive
end

function julia_isinteractive(x::Bool)
    global _julia_interactive = x
end

function __init__()
    py"""
    def _python_install(package):
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

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

    gap_fre_vector = [Float64(get(dict, '-', 0) / data_len) for dict in loc_dict_vector]

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
        if isgap_vector[ind] == true
            nogap_dis_vector = map(i -> Float64(dis_matrix_isgap_vector[i[1]] == true ? 1 : i[2]), enumerate(dis_matrix[:, id_vector[ind]]))
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
    get_data(R::AbstractRF, ami_arr::Int, excel_col::Char;
             norm::Bool=false, blosum::Int=62,
             sheet::String="Sheet1", title::Bool=true)

    get_data(R::AbstractRF, excel_col::Char;
             norm::Bool=false, blosum::Int=62,
             sheet::String="Sheet1", title::Bool=true)

# Examples
```julia-repl
julia> X, Y, L = get_data(R, 9, 'E');
```

Get data from `.fasta` file by converting selected BLOSUM matrix and `.xlsx` file at certain sheet and column.

# Arguments
- `R::AbstractRF` : for both [`RF`](@ref) and [`RFI`](@ref).
- `ami_arr::Int` : baseline for total number of mutations in samples at one location (when value is not determined, set to 1).
- `excel_col::Char` : column character for `.xlsx` file.
- `norm::Bool` : execute min-max normalization.
- `blosum::Int` : BLOSUM matrix number.
- `sheet::String` : `.xlsx` data sheet name
- `title::Bool` : when `.xlsx` have a header row, turn on `title`.

# Return
- `X::Matrix{Float64}` : independent variables data matrix.
- `Y::Vector{Float64}` : dependent variable data vector.
- `L::Vector{Tuple{Int, Char}}` : location tuple vector.  
    + `Int` : raw sequence index.
    + `Char` : criterion amino acid for corresponding location.
"""
function get_data(R::AbstractRF, ami_arr::Int, excel_col::Char; norm::Bool=false, blosum::Int=62, sheet::String="Sheet1", title::Bool=true)
    _get_data(R, ami_arr, excel_col, norm, blosum, sheet, title)
end

function get_data(R::AbstractRF, excel_col::Char; norm::Bool=false, blosum::Int=62, sheet::String="Sheet1", title::Bool=true)
    _get_data(R, 1, excel_col, norm, blosum, sheet, title)
end

function _get_data(R::AbstractRF, ami_arr::Int, excel_col::Char, norm::Bool, blonum::Int, sheet::String, title::Bool)
    data_len, loc_dict_vector, seq_matrix = _location_data(R.fasta_loc)
    blo = blosum_matrix(blonum)
    x_col_vector = Vector{Vector{Float64}}()
    loc_vector = Vector{Tuple{Int, Char}}()
    for (ind, (dict, col)) in enumerate(zip(loc_dict_vector, eachcol(seq_matrix)))
        max_val = maximum(values(dict))
        max_amino = _find_key(dict, max_val)
        if '-' ∉ keys(dict) && ami_arr ≤ data_len - max_val 
            push!(x_col_vector, [blo[max_amino, i] for i in col])
            push!(loc_vector, (ind, max_amino))
        end
    end

    excel_data = DataFrame(XLSX.readtable(R.data_loc, sheet, infer_eltypes=title)...)
    excel_select_vector = excel_data[!, Int(excel_col) - Int('A') + 1]
    if norm
        excel_select_vector = _min_max_norm(excel_select_vector)
    end
    
    x = Matrix{Float64}(hcat(x_col_vector...))
    y = Vector{Float64}(excel_select_vector)
    l = Vector{Tuple{Int, Char}}(loc_vector)
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
    return size(seq_matrix)[1], loc_vector, seq_matrix
end

function _find_key(dict::Dict{Char, Int}, tar::Int)
    for k in keys(dict)
        if dict[k] == tar
            return k
        end
    end
end

function _min_max_norm(vec::Vector{Float64})
    mi = minimum(vec)
    ma = maximum(vec)
    return [(i - mi) / (ma - mi) for i in vec]
end

"""
    get_amino_loc(R::AbstractRF, L::Vector{Tuple{Int, Char}})

Return string sequence index vector with start index.
"""
function get_amino_loc(R::AbstractRF, L::Vector{Tuple{Int, Char}})
    return [string(i[1] + R.amino_loc - 1) for i in L]
end

"""
    get_reg_importance(R::AbstractRF, X::Matrix{Float64}, Y::Vector{Float64},
                       L::Vector{Tuple{Int, Char}}, feat::Int, tree::Int;
                       val_mode::Bool=false, test_size::Float64=0.3,
                       show_number::Int=20, imp_iter::Int=60,
                       data_state::UInt64=@seed_u64,
                       learn_state::Int=@seed_i,
                       imp_state::UInt64=@seed_u64)

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
- `L::Vector{Tuple{Int, Char}}` : `L` data.
- `feat::Int` : number of selected features.
- `tree::Int` : number of trees.
- `val_mode::Bool` : when `val_mode` is true, function don't display anything.
- `test_size::Float64` : size of test set.
- `show_number::Int` : number of locations to show importance.
- `imp_iter::Int` : number of times to repeat to caculate a feature importance.
- `data_state::UInt64` : seed used to split data.
- `learn_state::Int` : seed used to caculate a regression model.
- `imp_state::UInt64` : seed used to caculate a feature importance.
"""
function get_reg_importance(R::AbstractRF, X::Matrix{Float64}, Y::Vector{Float64}, L::Vector{Tuple{Int, Char}}, feat::Int, tree::Int;
    val_mode::Bool=false, test_size::Float64=0.3, show_number::Int=20, imp_iter::Int=60,
    data_state::UInt64=@seed_u64, learn_state::Int=@seed_i, imp_state::UInt64=@seed_u64)
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, data_state=data_state)
    regr = RandomForestRegressor(n_trees=tree, n_subfeatures=feat, min_samples_leaf=1, rng=learn_state)
    DecisionTree.fit!(regr, x_train, y_train)

    if val_mode == false
        predict_test = parallel_predict(regr, x_test)
        predict_train = parallel_predict(regr, x_train)
        scatter(y_train, predict_train, color="orange", label="Train value")
        scatter(y_test, predict_test, color="blue" , label="Test value")
        legend(loc=4)
        xlabel("True Values")
        ylabel("Predictions")
        axis("equal")
        axis("square")
        xlim(0, xlim()[2])
        ylim(0, ylim()[2])
        plot([-1000, 1000], [-1000, 1000], color="black")
        PyPlot.title("Random Forest Regression Result")
        @show_pyplot
        @printf "NRMSE : %.6f\n" nrmse(predict_test, y_test)
    end
    return regr, _rf_importance(regr, DataFrame(X, get_amino_loc(R, L)), imp_iter, seed=imp_state, show_number=show_number, val_mode=val_mode)
end

"""
    rf_model(X::Matrix{Float64}, Y::Vector{Float64}, feat::Int, tree::Int;
             val_mode::Bool=false, test_size::Float64=0.3, 
             data_state::UInt64=@seed_u64, 
             learn_state::Int=@seed_i)

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
- `data_state::UInt64` : seed used to split data.
- `learn_state::Int` : seed used to caculate a regression model.
"""
function rf_model(X::Matrix{Float64}, Y::Vector{Float64}, feat::Int, tree::Int;
    val_mode::Bool=false, test_size::Float64=0.3, data_state::UInt64=@seed_u64, learn_state::Int=@seed_i)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, data_state=data_state)
    regr = RandomForestRegressor(n_trees=tree, n_subfeatures=feat, min_samples_leaf=1, rng=learn_state)
    DecisionTree.fit!(regr, x_train, y_train)

    if val_mode == false
        predict_test = parallel_predict(regr, x_test)
        predict_train = parallel_predict(regr, x_train)
        scatter(y_train, predict_train, color="orange", label="Train value")
        scatter(y_test, predict_test, color="blue" , label="Test value")
        legend(loc=4)
        PyPlot.title("Random Forest Regression Result")
        xlabel("True Values")
        ylabel("Predictions")
        axis("equal")
        axis("square")
        xlim(0, xlim()[2])
        ylim(0, ylim()[2])
        plot([-1000, 1000], [-1000, 1000], color="black")
        @show_pyplot
        @printf "NRMSE : %.6f\n" nrmse(predict_test, y_test)
    end
    return regr
end

"""
    rf_importance(R::AbstractRF, regr::RandomForestRegressor,
                  X::Matrix{Float64}, L::Vector{Tuple{Int, Char}};
                  val_mode::Bool=false,
                  show_number::Int=20, imp_iter::Int=60,
                  imp_state::UInt64=@seed_u64)
    
# Examples
```julia-repl
julia> F = rf_model(R, M, X, L, show_number=15);
```

Caculate feature importance for a target model, then draw feature importance list.

# Arguments
- `R::AbstractRF` : for both [`RF`](@ref) and [`RFI`](@ref).
- `regr::RandomForestRegressor` : target regression model.
- `X::Matrix{Float64}` : `X` data.
- `L::Vector{Tuple{Int, Char}}` : `L` data
- `val_mode::Bool` : when `val_mode` is true, function don't display anything.
- `show_number::Int` : number of locations to show importance.
- `imp_iter::Int` : number of times to repeat to caculate a feature importance.
- `imp_state::UInt64` : seed used to caculate a feature importance.
"""
function rf_importance(R::AbstractRF, regr::RandomForestRegressor, X::Matrix{Float64}, L::Vector{Tuple{Int, Char}};
    val_mode::Bool=false, show_number::Int=20, imp_iter::Int=60, imp_state::UInt64=@seed_u64)
    return _rf_importance(regr, DataFrame(X, get_amino_loc(R, L)), imp_iter, seed=imp_state, val_mode=val_mode, show_number=show_number)
end

function _rf_importance(regr::RandomForestRegressor, dx::DataFrame, iter::Int=60; 
                        seed::UInt64=@seed_u64, show_number::Int=20, val_mode::Bool=false)
    data_shap = ShapML.shap(explain = dx,
                    model = regr,
                    predict_function = _rf_dfpredict,
                    sample_size = iter,
                    seed = seed)
    data_plot = combine(groupby(data_shap, :feature_name), :shap_effect => x -> mean(abs.(x)))
    baseline = data_shap.intercept[1]
    feature_importance = data_plot[!, :shap_effect_function] / baseline
    if val_mode == false
        _view_importance(feature_importance, data_plot[!, :feature_name], baseline, show_number=show_number)
    end
    return feature_importance
end

function _rf_dfpredict(regr::RandomForestRegressor, x::DataFrame)
    return DataFrame(y_pred = parallel_predict(regr, Matrix{Float64}(x)))
end

function _view_importance(fe::Vector{Float64}, get_loc::Vector{String}, baseline::Float64; show_number::Int=20)
    sorted_idx = sortperm(fe, rev=true)
    bar_pos = [length(sorted_idx):-1:1;] .- 0.5
    barh(bar_pos[1:show_number], fe[sorted_idx][1:show_number], align="center")
    yticks(bar_pos[1:show_number], get_loc[sorted_idx][1:show_number])
    xlabel(@sprintf "|Shapley effect| (baseline = %.2f)" baseline)
    ylabel("Amino acid Location")
    PyPlot.title("Feature Importance - Mean Absolute Shapley Value")
    @show_pyplot
end

"""
    view_importance(R::AbstractRF, L::Vector{Tuple{Int, Char}},
                    F::Vector{Float64}; show_number::Int=20)

# Examples
```julia-repl
julia> view_importance(R, L, F);

julia> view_importance(R, L, MF, show_number=30);
```

Draw feature importance list.

# Arguments
- `R::AbstractRF` : for both [`RF`](@ref) and [`RFI`](@ref).
- `L::Vector{Tuple{Int, Char}}` : `L` data.
- `F::Vector{Float64}` : feature importance vector.
- `show_number::Int` : number of locations to show importance.
"""
function view_importance(R::AbstractRF, L::Vector{Tuple{Int, Char}}, F::Vector{Float64}; show_number::Int=20)
    _view_importance(F, get_amino_loc(R, L), show_number=show_number)
end

function _view_importance(fi::Vector{Float64}, get_loc::Vector{String}; show_number::Int=20)
    fi /= maximum(fi)
    sorted_idx = sortperm(fi, rev=true)
    bar_pos = [length(sorted_idx):-1:1;] .- 0.5
    barh(bar_pos[1:show_number], fi[sorted_idx][1:show_number], align="center")
    yticks(bar_pos[1:show_number], get_loc[sorted_idx][1:show_number])
    xlabel("Feature Importance")
    ylabel("Amino acid Location")
    PyPlot.title("Relative Mean Absolute Shapley Value")
    @show_pyplot
end

"""
    iter_get_reg_importance(R::AbstractRF, X::Matrix{Float64}, Y::Vector{Float64},
                            L::Vector{Tuple{Int, Char}},
                            feet::Int, tree::Int, iter::Int;
                            val_mode::Bool=false, test_size::Float64=0.3,
                            show_number::Int=20, imp_iter::Int=60,
                            data_state::UInt64=@seed_u64, imp_state::UInt64=@seed_u64)

# Examples
```julia-repl
julia> MF, SF = iter_get_reg_importance(R, X, Y, L, 3, 700, 10);
```
Calculate feature importance by repeating `iter::Int` times with a fixed data and importance seed, then draw feature importance list inclding standard deviation.

Returns the mean and standard deviation of feature importance.

# Arguments
- `R::AbstractRF` : for both [`RF`](@ref) and [`RFI`](@ref).
- `X::Matrix{Float64}` : `X` data.
- `Y::Vector{Float64}` : `Y` data.
- `L::Vector{Tuple{Int, Char}}` : `L` data.
- `feat::Int` : number of selected features.
- `tree::Int` : number of trees.
- `val_mode::Bool` : when `val_mode` is true, function don't display anything.
- `test_size::Float64` : size of test set.
- `show_number::Int` : number of locations to show importance.
- `imp_iter::Int` : number of times to repeat to caculate a feature importance.
- `data_state::UInt64` : seed used to split data.
- `imp_state::UInt64` : seed used to caculate a feature importance.
- `learn_state_seed::UInt64` : seed used to generate seed used to caculate a regression model.
"""
function iter_get_reg_importance(R::AbstractRF, X::Matrix{Float64}, Y::Vector{Float64}, L::Vector{Tuple{Int, Char}}, feet::Int, tree::Int, iter::Int;
    val_mode::Bool=false, test_size::Float64=0.3, show_number::Int=20, imp_iter::Int=60,
    data_state::UInt64=@seed_u64, imp_state::UInt64=@seed_u64, learn_state_seed::UInt64=@seed_u64)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, data_state=data_state)
    f = zeros(length(L), iter)
    n = zeros(iter)
    loc_list = get_amino_loc(R, L)
    learn_state_vector = Vector{Int}(rand(MersenneTwister(learn_state_seed), 0:typemax(Int), iter))
    for i in 1:iter
        f[:, i], n[i] = _iter_get_reg_importance(X, x_train, x_test, y_train, y_test, loc_list, feet, tree, imp_iter, imp_state, learn_state_vector[i])
    end
    mf, sf = mean(f, dims=2)[:, 1], std(f, dims=2)[:, 1]

    if val_mode == false
        _iter_view_importance(mf, sf, loc_list, show_number=show_number)
        @printf "NRMSE : %.6f\n" mean(n)
    end

    return mf, sf
end

function _iter_get_reg_importance(x::Matrix{Float64}, x_train::Matrix{Float64}, x_test::Matrix{Float64}, y_train::Vector{Float64}, y_test::Vector{Float64}, loc::Vector{String}, feet::Int, tree::Int, imp_iter::Int, imp_state::UInt64, learn_state::Int)
    regr = RandomForestRegressor(n_trees=tree, n_subfeatures=feet, min_samples_leaf=1, rng=learn_state)
    DecisionTree.fit!(regr, x_train, y_train)
    return _rf_importance(regr, DataFrame(x, loc), imp_iter, seed=imp_state, val_mode=true), nrmse(regr, x_test, y_test)
end

"""
    view_importance(R::AbstractRF, L::Vector{Tuple{Int, Char}},
                    MF::Vector{Float64}, SF::Vector{Float64};
                    show_number::Int=20)

# Examples
```julia-repl
julia> view_importance(R, L, MF, SF, show_number=30);
```

Draw feature importance list with standard deviation.

# Arguments
- `R::AbstractRF` : for both [`RF`](@ref) and [`RFI`](@ref).
- `L::Vector{Tuple{Int, Char}}` : `L` data.
- `MF::Vector{Float64}` : mean feature importance vector.
- `SF::Vector{Float64}` : standard deviation feature importance vector.
- `show_number::Int` : number of locations to show importance.
"""
function view_importance(R::AbstractRF, L::Vector{Tuple{Int, Char}}, MF::Vector{Float64}, SF::Vector{Float64}; show_number::Int=20)
    _iter_view_importance(MF, SF, get_amino_loc(R, L), show_number=show_number)
end

function _iter_view_importance(fe::Vector{Float64}, err::Vector{Float64}, loc::Vector{String}; show_number::Int=20)
    norm_val = maximum(fe)
    fe /= norm_val
    err /= norm_val
    sorted_idx = sortperm(fe, rev=true)
    bar_pos = [length(sorted_idx):-1:1;] .- 0.5
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
                  data_state::UInt64=@seed_u64,
                  learn_state::Int=@seed_i)

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
- `data_state::UInt64` : seed used to split data.
- `learn_state::Int` : seed used to caculate a regression model.
"""
function get_reg_value(RI::AbstractRFI, X::Matrix{Float64}, Y::Vector{Float64};
    val_mode::Bool=false, test_size::Float64=0.3, data_state::UInt64=@seed_u64, learn_state::Int=@seed_i)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, data_state=data_state)
    z = zeros(Float64, length(RI.nfeat), length(RI.ntree))
    task = [(i[1], j[1], i[2], j[2]) for i in enumerate(RI.nfeat), j in enumerate(RI.ntree)]
    
    Threads.@threads for (i, j, feat, tree) in task
        z[i,  j] = _get_reg_value(x_train, x_test, y_train, y_test, feat, tree, learn_state)
    end

    if val_mode == false
        view_reg3d(RI, z, title="NRMSE value")
    end
    return z
end

function _get_reg_value(x_train::Matrix{Float64}, x_test::Matrix{Float64}, y_train::Vector{Float64}, y_test::Vector{Float64}, feat::Int, tree::Int, learn_state::Int)
    regr = RandomForestRegressor(n_trees=tree, n_subfeatures=feat, min_samples_leaf=1, rng=learn_state)
    DecisionTree.fit!(regr, x_train, y_train)
    return nrmse(regr, x_test, y_test)
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
    nfeat_list = [RI.nfeat;]' .* ones(length(RI.ntree))
    ntree_list = ones(length(RI.nfeat))' .* [RI.ntree;]
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
                       learn_state::Int=@seed_i,
                       data_state_seed::UInt64=@seed_u64)

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
- `learn_state::Int` : seed used to caculate a regression model.
- `data_state_seed::UInt64` : seed used to generate seed used to split data.
"""
function iter_get_reg_value(RI::AbstractRFI, X::Matrix{Float64}, Y::Vector{Float64}, iter::Int;
    val_mode::Bool=false, test_size::Float64=0.3, learn_state::Int=@seed_i, data_state_seed::UInt64=@seed_u64)
    z = zeros(length(RI.nfeat), length(RI.ntree), iter)
    data_state_vector = Vector{UInt64}(rand(MersenneTwister(data_state_seed), UInt64, iter))
    for i = 1:iter
        z[:, :, i] = get_reg_value(RI, X, Y, val_mode=true, test_size=test_size, data_state=data_state_vector[i], learn_state=learn_state)
    end

    vz, sz = mean(z, dims=3)[:, :, 1], std(z, dims=3)[:, :, 1]

    if val_mode == false
        view_reg3d(RI, vz, title="NRMSE value", scale=2)
        view_reg3d(RI, sz, title="NRMSE SD value", scale=3)
    end

    return vz, sz
end

end # module end