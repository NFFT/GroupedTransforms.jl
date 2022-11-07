abstract type GroupedCoefficients end

@doc raw"""
    GroupedCoefficientsComplex

A struct to hold complex coefficients belonging to indices in a grouped index set

```math
    \mathcal{I}_{\pmb{N}}(U) = \left\{ \pmb{k} \in \Z^d : \mathrm{supp} \pmb{k} \in U, \pmb{k}_{\mathrm{supp} \pmb{k}} \in [- \frac{N_{\mathrm{supp} \pmb{k} }}{2}, \frac{N_{ \mathrm{supp} \pmb{k} }}{2} - 1 ) \right\}.
```

# Fields
* `setting::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{Bool}}}}` - uniquely describes the setting such as the bandlimits ``N_{\pmb u}``, see also [`get_setting(system::String,d::Int,ds::Int,N::Vector{Int},dcos::Vector{Bool})::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{Bool}}}}`](@ref) and [`get_setting(system::String,U::Vector{Vector{Int}},N::Vector{Int},dcos::Vector{Bool})::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{Bool}}}}`](@ref)
* `data::Union{Vector{ComplexF64},Nothing}` - the vector of coefficients

# Constructor
    GroupedCoefficientsComplex( setting, data = nothing )

# Additional Constructor
    GroupedCoefficients( setting, data = nothing )
"""
struct GroupedCoefficientsComplex <: GroupedCoefficients
    setting::Vector{
        NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{Bool}}}
    }
    data::Vector{ComplexF64}

    function GroupedCoefficientsComplex(
        setting,
        data::Union{Vector{ComplexF64},Nothing} = nothing,
    )
        try
			println("test")
			for s = setting
				println(s[:bandwidths])
				println(s[:mode].datalength(s[:bandwidths]))
			end
            N = sum(s -> s[:mode].datalength(s[:bandwidths]), setting)
            println(N)
			if isnothing(data)
                data = zeros(ComplexF64, N)
            end
            if length(data) != N
                error("the supplied data vector has the wrong length.")
            end
            return new(setting, data)
        catch
            error("The mode is not supportet yet or does not have the function datalength.")
        end
    end
end

@doc raw"""
    GroupedCoefficientsReal

A struct to hold real valued coefficients belonging to indices in a grouped index set

```math
    \mathcal{I}_{\pmb{N}}(U) = \left\{ \pmb{k} \in \Z^d : \mathrm{supp} \pmb{k} \in U, \pmb{k}_{\mathrm{supp} \pmb{k}} \in [0, N_{\mathrm{supp} \pmb{k} } - 1 ] \right\}.
```

# Fields
* `setting::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{Bool}}}}` - uniquely describes the setting such as the bandlimits ``N_{\pmb u}``, see also [`get_setting(system::String,d::Int,ds::Int,N::Vector{Int},dcos::Vector{Bool})::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{Bool}}}}`](@ref) and [`get_setting(system::String,U::Vector{Vector{Int}},N::Vector{Int},dcos::Vector{Bool})::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{Bool}}}}`](@ref)
* `data::Union{Vector{Float64},Nothing}` - the vector of coefficients

# Constructor
    GroupedCoefficientsReal( setting, data = nothing )

# Additional Constructor
    GroupedCoefficients( setting, data = nothing )
"""
struct GroupedCoefficientsReal <: GroupedCoefficients
    setting::Vector{
        NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{Bool}}}
    }
    data::Vector{Float64}

    function GroupedCoefficientsReal(
        setting,
        data::Union{Vector{Float64},Nothing} = nothing,
    )
        try
            N = sum(s -> s[:mode].datalength(s[:bandwidths]), setting)
            if isnothing(data)
                data = zeros(Float64, N)
            end
            if length(data) != N
                error("the supplied data vector has the wrong length.")
            end
            return new(setting, data)
        catch
            error("The mode is not supportet yet or does not have the function datalength.")
        end
    end
end

function GroupedCoefficients(
    setting::Vector{
        NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{Bool}}}
    },
    data::Union{Vector{ComplexF64},Vector{Float64},Nothing} = nothing,
)
    if (setting[1][:mode] == NFFTtools || setting[1][:mode] == NFFCTtools)
        return GroupedCoefficientsComplex(setting, data)
    elseif (setting[1][:mode] == NFCTtools  || setting[1][:mode] == CWWTtools )
        return GroupedCoefficientsReal(setting, data)
    end
end

@doc raw"""
    fhat::GroupedCoefficients[u::Vector{Int}]

This function overloads getindex of GroupedCoefficients such that you can do `fhat[[1,3]]` to obtain the basis coefficients of the corresponding ANOVA term defined by `u`.
"""
function Base.:getindex(fhat::GroupedCoefficients, u::Vector{Int})::Vector{<:Number}
    start = 1
    for s in fhat.setting
        if s[:u] == u
            stop = start + s[:mode].datalength(s[:bandwidths]) - 1
            return fhat.data[start:stop]
        else
            start += s[:mode].datalength(s[:bandwidths])
        end
    end
    error("This term is not contained")
end

@doc raw"""
    fhat::GroupedCoefficients[idx::Int]

This function overloads getindex of GroupedCoefficients such that you can do `fhat[1]` to obtain the basis coefficient determined by `idx`.
"""
function Base.:getindex(fhat::GroupedCoefficients, idx::Int64)::Number
    return fhat.data[idx]
end

@doc raw"""
    fhat::GroupedCoefficients[u::Vector{Int}] = fhatu::Union{Vector{ComplexF64},Vector{Float64}}

This function overloads setindex of GroupedCoefficients such that you can do `fhat[[1,3]] = [1 2 3]` to set the basis coefficients of the corresponding ANOVA term defined by `u`.
"""
function Base.:setindex!(
    fhat::GroupedCoefficients,
    fhatu::Union{Vector{ComplexF64},Vector{Float64}},
    u::Vector{<:Integer},
)
    if (isa(fhat, GroupedCoefficientsComplex) && isa(fhatu, Vector{Float64})) ||
       (isa(fhat, GroupedCoefficientsReal) && isa(fhatu, Vector{ComplexF64}))
        error("Type mismatch.")
    end
    start = 1
    for s in fhat.setting
        if s[:u] == u
            stop = start + s[:mode].datalength(s[:bandwidths]) - 1
            fhat.data[start:stop] = fhatu
            return
        else
            start += s[:mode].datalength(s[:bandwidths])
        end
    end
    error("This term is not contained")
end

@doc raw"""
    fhat::GroupedCoefficients[idx::Int] = z::Number

This function overloads setindex of GroupedCoefficients such that you can do `fhat[1] = 3` to set the basis coefficient determined by `idx`.
"""
function Base.:setindex!(fhat::GroupedCoefficients, z::Number, idx::Int64)
    fhat.data[idx] = z
end

@doc raw"""
    vec( fhat::GroupedCoefficients )::Vector{<:Number}

This function returns the vector of the basis coefficients of fhat. This is useful for working with `lsqr` or similar.
"""
Base.:vec(fhat::GroupedCoefficients)::Vector{<:Number} = fhat.data

@doc raw"""
    *( z::Number, fhat::GroupedCoefficients )::GroupedCoefficients

This function defines the multiplication of a number with a GroupedCoefficients object.
"""
Base.:*(α::Number, fhat::GroupedCoefficients) =
    GroupedCoefficients(fhat.setting, α * vec(fhat))

@doc raw"""
    +( z::Number, fhat::GroupedCoefficients )::GroupedCoefficients

This function defines the addition of two GroupedCoefficients objects.
"""
function Base.:+(fhat::GroupedCoefficients, ghat::GroupedCoefficients)::GroupedCoefficients
    if fhat.setting == ghat.setting
        return GroupedCoefficients(fhat.setting, vec(fhat) + vec(ghat))
    else
        error("Settings mismatch.")
    end
end

@doc raw"""
    -( z::Number, fhat::GroupedCoefficients )::GroupedCoefficients

This function defines the subtraction of two GroupedCoefficients objects.
"""
Base.:-(fhat::GroupedCoefficients, ghat::GroupedCoefficients) = fhat + (-1 * ghat)

function set_data!(
    fhat::GroupedCoefficients,
    data::Union{Vector{ComplexF64},Vector{Float64}},
)
    if (isa(fhat, GroupedCoefficientsComplex) && isa(data, Vector{Float64})) ||
       (isa(fhat, GroupedCoefficientsReal) && isa(data, Vector{ComplexF64}))
        error("Type mismatch.")
    end
    (fhat.data)[:] = data
end

function norms(fhat::GroupedCoefficients; dict =false)::Union{Vector{Float64},Dict{Vector{Int},Float64}}
    setting = fhat.setting
	if dict == false
    	return [norm(fhat[setting[i][:u]]) for i = 1:length(setting)]
	else
		dd = Dict{Vector{Int},Float64}()
		for i  = 1: length(fhat.setting)
			if fhat.setting[i][:u] != []
				dd[fhat.setting[i][:u]] = norm(fhat[setting[i][:u]])
			end
		end
		return dd
	end
end


function norms(fhat::GroupedCoefficients, m::Int ; dict =false)::Union{Vector{Float64},Dict{Vector{Int},Float64}}
	setting = fhat.setting
	if dict == false
	    if setting[1][:mode] == CWWTtools
	        n = zeros(length(fhat.setting))
			for i  = 1: length(fhat.setting)
				s = fhat.setting[i]
				d = length(fhat.setting[i].u)
				ac_in = 1    #actual index
				freq = CWWTtools.cwwt_index_set(fhat.setting[i].bandwidths)
				if d == 1
					freq = transpose(freq)
				end
				for jj in 1:size(freq,2)
					j = freq[:,jj]
					a = fhat[s.u][ac_in:ac_in+2^sum(j)-1]
					Psi = Circulant(variances(j[1],m))
						if d>1
							for dd =2:d
								Psi = kron(Psi,Circulant(variances(j[dd],m)))
							end
						end
					n[i] = n[i] + a'*Psi*a
					ac_in = ac_in +2^sum(j)
				end
			end
			return sqrt.(n)
	    else
	        return [norm(fhat[setting[i][:u]]) for i = 1:length(setting)]
	    end
	else #dict ==true
		dd = Dict{Vector{Int},Float64}()
	    if setting[1][:mode] == CWWTtools
	        n = zeros(length(fhat.setting))
			for i  = 1: length(fhat.setting)
				s = fhat.setting[i]
				d = length(fhat.setting[i].u)
				ac_in = 1    #actual index
				freq = CWWTtools.cwwt_index_set(fhat.setting[i].bandwidths)
				if d == 1
				freq = transpose(freq)
				end
				for jj in 1:size(freq,2)
					j = freq[:,jj]
					a = fhat[s.u][ac_in:ac_in+2^sum(j)-1]
					Psi = Circulant(variances(j[1],m))
						if d>1
							for dd =2:d
								Psi = kron(Psi,Circulant(variances(j[dd],m)))
							end
						end
					n[i] = n[i] + a'*Psi*a
					ac_in = ac_in +2^sum(j)
				end
				if fhat.setting[i].u != []
					dd[fhat.setting[i].u] = sqrt(n[i])
				end
			end
			return dd
	    else
			for i  = 1: length(fhat.setting)
	            if fhat.setting[i][:u] != []
	                dd[fhat.setting[i][:u]] = norm(fhat[setting[i][:u]])
	            end
			end
	        return dd
	    end
	end  #dict
end  #function




function norms(fhat::GroupedCoefficients, what::GroupedCoefficients)::Vector{Float64}
    c = GroupedCoefficients(fhat.setting, (sqrt.(real.(what.data))) .* fhat.data)
    setting = c.setting
    return [norm(c[setting[i][:u]]) for i = 1:length(setting)]
end


"""matrix of variances between two basis functions, needed for wavelet basis, since they are not orthonormal"""
function variances(j::Int,m::Int)::Vector{Float64}
"""
INPUT
j 		... level of wavelet
m 		...	order of wavelet
OUTPUT
y = (<psi_{j,0},psi_{j,k}>)for k = 0...2^j-1
								(psi_{j,k}) are the wavelets, output contains a vector of all scalar products of one level
								for 2^j >2m*1 the same values, but more zeros for higher j.

"""
if m ==2
	if j == 0
		y = [1/3]
	elseif j == 1
		y = [0.240740740740715, 0.092592592592576]
	elseif j ==2
		y = [0.250000000000097,0.046296296296494,-0.009259259259238, 0.046296296296451]
	else
		y = zeros(2^j)
		y[1:3]= [0.249999999999968, 0.046296296296379, -0.004629629629584]
		y[end:-1:end-1] = [ 0.046296296296379,-0.004629629629584]
	end

elseif m ==3
	if j == 0
		y = [0.133333333333334]
	elseif j == 1
		y = [0.085629629629627, 0.047703703703701]
	elseif j == 2
		y = [0.098444444444468,0.023851851851835,-0.012814814814815, 0.023851851851891]
	elseif j == 3
		y= [0.098443287037094,0.024151620370357,-0.006407407407473,-2.997685185108751e-04,1.157407410282006e-06,-2.997685185108751e-04,-0.006407407407473,0.024151620370]
	else
		y = zeros(2^j)
		y[1:5]= [0.098443287037052, 0.024151620370458, -0.006407407407375, -2.997685186147822e-04,5.787036288131668e-07]
		y[end:-1:end-3] = y[2:1:5]
	end

elseif m ==4
	if j == 0
		y = [0.053968253968254]
	elseif j == 1
		y = [0.032014093350, 0.021954160617785]
	elseif j == 2
		y = [0.041543521817902,0.010977080308839,-0.009529428467484, 0.010977080308902]
	elseif j == 3
		y= [0.041534149423629,0.011629855618373,-0.004764714233735,-6.527753094869201e-04,9.372394228547530e-06,-6.527753094788768e-04,-0.004764714233719,0.011629855618379]
	else
		y = zeros(2^j)
		y[1:7]= [0.041534149423653, 0.011629855618368, -0.004764714225921, -6.528682451447852e-04,4.686197131487609e-06, 9.293564145017154e-08,-7.821712475559538e-12]
		y[end:-1:end-5] = y[2:1:7]
	end
end
return y
end
