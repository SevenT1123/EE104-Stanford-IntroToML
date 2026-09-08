import JSON
import LinearAlgebra: Adjoint, Transpose

#===============================================================================
| readclassjson(filename)
| takes the name of a JSON data file as input, and returns a dictionary
| containing the variables defined in the data file
===============================================================================#
function readclassjson(filename)
    #===========================================================================
    | create a dictionary of data types
    ===========================================================================#
    datatype_strings = Dict(
        "Int64" => Int64,
        "Int32" => Int32,
        "Int8" => Int8,
        "UInt64" => UInt64,
        "UInt32" => UInt32,
        "UInt8" => UInt8,
        "Float64" => Float64,
        "Float32" => Float32,
        "String" => String,
        "Bool" => Bool)

    #===========================================================================
    | parse the JSON file
    ===========================================================================#
    raw_data = JSON.parsefile(filename)

    #===========================================================================
    | format the data
    ===========================================================================#
    formatted_data = Dict()
    for varname in keys(raw_data)
        #=======================================================================
        | get the type of the data
        =======================================================================#
        vartype = datatype_strings[raw_data[varname]["type"]]

        #=======================================================================
        | convert the data
        =======================================================================#
        vardata = raw_data[varname]["data"]
        if isa(vardata , Array)
            vartype = Array{vartype}
        end
        if isa(vardata[1] , Array) 
            vardata = permutedims(hcat(vardata...))
        end
        vardata = convert(vartype, vardata)
        if "juliatype" in keys(raw_data[varname])
            juliatype = raw_data[varname]["juliatype"]
            T = eval(Meta.parse(juliatype))
            vardata = maketype(T, vardata)
        end
        
        #=======================================================================
        | store the converted data
        =======================================================================#
        formatted_data[varname] = vardata
    end

    return formatted_data
end

function maketype(a::Type{T}, x) where {T <: Transpose{X, Matrix{X}}} where {X}
    # for Transpose{Float64, Matrix{Float64}}
    # Matrix(transpose(X)) creates the transpose
    # by rearranging the data.
    # Then calling transpose creates a transpose object.
    return transpose(Matrix(transpose(x)))
end

function maketype(a::Type{T}, x) where {T <: Transpose{X, Vector{X}}} where {X}
    # for Transpose{Int8, Vector{Int8}}
    return transpose(x[:])
end

function maketype(a::Type{T}, x) where {T <: Adjoint{X, Matrix{X}}} where {X}
    # for Adjoint{UInt64, Matrix{UInt64}}
    return Matrix(x')'
end

function maketype(a::Type{T}, x) where {T <: Adjoint{X, Vector{X}}} where {X}
    # for Adjoint{Float64, Vector{Float64}}
    return x[:]'
end

function maketype(a::Type{T}, x) where {T <: UnitRange{X}} where {X}
    return UnitRange(x[1],x[2])
end

function maketype(a::Type{T}, x) where {T <: StepRange{X}} where {X}
    return StepRange(x[1],x[2],x[3])
end


maketype(T, x) = x