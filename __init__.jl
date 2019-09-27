using PyCall, PyPlot
plt = pyimport("matplotlib.pyplot")

function plot_loss_diagram(; labels_inside=false)
    grid = -2:0.004:2
    plt.figure()
    plt.xlabel("y_iw^T x_i", fontsize=18)
    # plt.xlabel('raw model output')
    plt.ylabel("f_i(w)", fontsize=18)
    plt.xlim(-2,2)
    plt.ylim(-0.025,3)
    plt.fill_between([0, 2], -1, 3, facecolor="blue", alpha=0.2);
    plt.fill_between([-2, 0], -1, 3, facecolor="red", alpha=0.2);
    plt.yticks([0,1,2,3]);

    if labels_inside
        plt.text(-1.95, 2.73, "incorrect prediction", fontsize=15) # 2.68
        plt.text(0.15, 2.73, "correct prediction", fontsize=15)
    else
        plt.text(-1.95, 3.1, "incorrect prediction", fontsize=15) # 2.68
        plt.text(0.15, 3.1, "correct prediction", fontsize=15)
    end

    plt.tight_layout()
end
function plot_contours(ax, clf, xx, yy, alpha; proba=true, transformation=nothing, cmap=plt.cm.coolwarm, params...)
    #=Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    =#

    X = hcat(xx[:], yy[:])
    if transformation != nothing
        X = transformation(X)
    end

    if proba == "raw"
        Z = clf.decision_function(X)
        Z = reshape(Z,size(xx))
        # out = ax.contourf(xx, yy, Z, **params)
        out = ax.imshow(Z,extent=(minimum(xx), maximum(xx), minimum(yy), maximum(yy)), origin="lower", cmap=cmap, params...)
        ax.contour(xx, yy, Z, levels=[0])
    elseif proba
        Z = clf.predict_proba(X)[:,end-1]
        Z = reshape(Z,size(xx))
        out = ax.imshow(Z,extent=(minimum(xx), maximum(xx), minimum(yy), maximum(yy)), origin="lower", vmin=0, vmax=1, aspect="auto", params...)
        ax.contour(xx, yy, Z, levels=[0.5])
    else
        Z = clf.predict(X)
        Z = reshape(Z,size(xx))
        out = ax.contourf(xx, yy, Z, alpha=alpha, cmap=cmap, params...)
    end

    return out
end
function make_meshgrid(x, y; num_pts=300, lims=nothing)
    #=
    Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    =#

    if lims == nothing
        x_min, x_max = minimum(x) - 1, maximum(x) + 1
        y_min, y_max = minimum(y) - 1, maximum(y) + 1
    else
        x_min, x_max, y_min, y_max = lims
    end
    xx = repeat(range(x_min,stop=x_max,length=num_pts)',num_pts,1)
    yy = repeat(range(y_min,stop=y_max,length=num_pts),1,num_pts)
    
    return xx, yy
end
# adapted from http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
function plot_classifier(X, y, clf; ax=nothing, ticks=false, proba=false, lims=nothing, transformation=nothing, show_data=true, gray_photocopy=false, proba_showtitle=true, cmap=nothing,  kwargs...) # assumes classifier "clf" is already fit
    X0, X1 = X[:, 1], X[:, 2]
    xx, yy = make_meshgrid(X0, X1, lims=lims)

    if ax == nothing
        plt.figure()
        ax = plt.gca()
        show = true
    else
        show = false
    end

    kwargs=Dict{Symbol,Any}(kwargs)
    if gray_photocopy
        kwargs[:cmap] = get(kwargs,:cmap,plt.cm.YlOrRd) # default cmap for photocopied grayscale exams
    else
        kwargs[:cmap] = get(kwargs,:cmap,plt.cm.coolwarm) # default cmap, but user can overrule it
    end

    # can abstract some of this into a higher-level function for learners to call
    cs = plot_contours(ax, clf, xx, yy, 0.8, cmap=get(kwargs,:cmap,nothing), proba=proba, transformation=transformation)

    if proba == "raw"
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel("raw model output", fontsize=20, rotation=270, labelpad=30)
        cbar.ax.tick_params(labelsize=14)
    elseif proba
        cbar = plt.colorbar(cs)
        if proba_showtitle
            cbar.ax.set_ylabel("probability of red Delta class", fontsize=20, rotation=270, labelpad=30)
        end
        cbar.ax.tick_params(labelsize=14)
    end
    if show_data
        #ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=30, edgecolors='k', linewidth=1)
        labels = unique(y)
        if length(labels) == 2
            ax.scatter(X0[y.==labels[1]], X1[y.==labels[1]], s=60, c=:b, marker=:o, edgecolors=:k)
            ax.scatter(X0[y.==labels[2]], X1[y.==labels[2]], s=60, c=:r, marker="^", edgecolors=:k)
        end
        if length(labels) == 3
            ax.scatter(X0[y.==labels[1]], X1[y.==labels[1]], s=60, c=:b, marker=:o, edgecolors=:k)
            ax.scatter(X0[y.==labels[2]], X1[y.==labels[2]], s=60, c=:r, marker='^', edgecolors=:k)
            ax.scatter(X0[y.==labels[3]], X1[y.==labels[3]], s=60, c=:k, marker=:x, edgecolors=:k)
        else
            ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=50, edgecolors=:k, linewidth=1)
            # plt.legend(labels) # doesn't work
            # see https://stackoverflow.com/questions/43967663/scatter-plot-with-legend-colored-by-group-without-multiple-calls-to-plt-scatter
        end
    end

    ax.set_xlim(minimum(xx), maximum(xx))
    ax.set_ylim(minimum(yy), maximum(yy))
#     ax.set_xlabel(data.feature_names[0])
#     ax.set_ylabel(data.feature_names[1])
    if !ticks
        ax.set_xticks(())
        ax.set_yticks(())
    end
#     ax.set_title(title)
    # if show:
        # plt.show()
    # else:
    return ax
end
using LinearAlgebra
function eye(n::Int64)
    return Array{Int64}(I,n,n)
end
using MLDataPattern
function train_test_split(X,y;at::Float64=0.5)
    (X,y) = shuffleobs((X',y))
    (xtrain,ytrain),(xvalid,yvalid) = splitobs((X,y),at=at)
    return (xtrain',xvalid',ytrain,yvalid)
end

mutable struct PolynomialFeaturesModel
    transform
    n_input_features :: Int64
    n_output_features :: Int64
end
function countCombinations(nFeatures)
    result = 1
    for i in 1:nFeatures-1
        result += binomial(nFeatures,i)
    end
    return result
end
function initializePolynomialFeatures(X;degree::Int64=2,interaction_only::Bool=false,include_bias::Bool=true)
    n_samples,n_features = size(X)
    n_input_features = n_features
    n_output_features = include_bias ? 1 : 0
    n_output_features += n_features*degree + countCombinations(n_features)
    function transform(X)
        n_samples,n_features = size(X)
        if (n_features != n_input_features)
            throw(DimensionMismatch("X shape does not match training shape"))
        end
        result = include_bias ? ones(n_samples,1) : []
        for i in 1:degree
            for j in 1:n_features
                result = hcat(result,X[:,j] .^ i)
            end                    
        end
        for i in 2:n_features
            indices = collect(combinations(1:n_features,i))
            for j in indices
                nxt = [prod(X[k,j]) for k in 1:n_samples]
                result = hcat(result,nxt)
            end
        end
        return result
    end
    return PolynomialFeaturesModel(transform,n_input_features,n_output_features)
end
function allClose(a,b)
    all(broadcast(abs,a - b) .<= (1e-08 .+ 1e-05 .* broadcast(abs,b)))
end
function euclidean_dist_squared(X, Xtest)
    (n,d) = size(X)
    (t,d2) = size(Xtest)
    @assert(d==d2)
    return X.^2*ones(d,t) + ones(n,d)*(Xtest').^2 - 2X*Xtest'
end
function RBF_features(Xtrain, Xtest; σ=1)
    return broadcast(exp,-0.5.*euclidean_dist_squared(Xtest,Xtrain)./σ^2)
end
function conv(u,v;mode="full")
    nu = length(u)
    nv = length(v)
    if nu==0||nv==0
        throw( DomainError("parameter u or v",
            "Argument vectors are supposed to be non-empty.") )
    elseif nv>nu
        u, v = v, u
        nu, nv = nv, nu
    end
    if mode=="full"
        n = nu+nv-1
        return [u[max(1, i+1-nv):min(i,nu)]'*v[i<nv ? 
                    (i:-1:1) : (end:-1:max(1,(i)-(n-nv)))]
                    for i in 1:n]
    elseif mode=="same"
        n = nu+nv-1
        res_n = max(nu,nv)
        start = div(n-res_n,2)+1
        return [u[max(1, start+i-nv):min(start+i-1,nu)]'*v[start+i-1<nv ? 
                    (start+i-1:-1:1) : (end:-1:max(1,(start+i-1)-(n-nv)))] 
                    for i in 1:res_n]
    elseif mode=="valid"
        n = nu-nv+1
        if n < 1
            return []
        end
        return [u[i:i+nv-1]'*v[end:-1:1] for i in 1:n]
    else
        throw(DomainError("\"$mode\"", "Parameter mode is not valid."))
    end
end