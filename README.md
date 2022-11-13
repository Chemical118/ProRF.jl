# ProRF.jl <img src="docs/src/assets/logo.svg" alt="ProRF logo" align="right" height="160" style="display: inline-block;">
`ProRF` provides a full process for applying the random forest model of protein sequences using `DecisionTree`.

## Install
You must install Python module `Bokeh` to preprocess raw sequence. For more information, see `PyCall` [main page](https://github.com/JuliaPy/PyCall.jl).
```julia
using Pkg
Pkg.add(url="https://github.com/Chemical118/ProRF.jl")
```

## Documentaion
[ProRF.jl Documentaion](https://chemical118.github.io/ProRF.jl/dev/)  

## Package
This is an environment where you can run `ProRF` by using `Docker`
```bash
$ docker pull ghcr.io/chemical118/prorf:latest
```

`ProRF` is still developing! Please send email to <wowo0118@korea.ac.kr> if you have any question.