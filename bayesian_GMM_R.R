# generate from mixture of normals
#' @param n number of samples
#' @param pi mixture proportions
#' @param mu mixture means
#' @param s mixture standard deviations
rmix = function(n,pi,mu,s){
  z = sample(1:length(pi),prob=pi,size=n,replace=TRUE)
  X = rbind(mvtnorm::rmvnorm(sum(z==1), mu[,1], s[[1]]), mvtnorm::rmvnorm(sum(z==2), mu[,2], s[[2]])) # , s[[z]]
  Z = c(rep(1, sum(z==1)), rep(2, sum(z==2)))
  #x = rnorm(n,mu[z],s[z])
  return(list(X,Z))
}
out = rmix(n=1000,pi=c(0.3,0.7),matrix(c(-3.5, 1.0, 6.0, 3.0), byrow=TRUE, ncol=2),
         s=list(matrix(c(1.5, -.2, -.2, 1.5), ncol=2,byrow=TRUE), 
                matrix(c(2, 1, 1, 2), byrow=TRUE, ncol=2)))
x = out[[1]]
z = out[[2]]
iz = out[[2]]
hist(x)
library(ggplot2)
df <- data.frame(x1=x[,1], x2=x[,2], z=as.factor(z))
ggplot(df, aes(x=x1, y=x2, color=z))+
  geom_point()

sample_mvn_mu <- function(x, pi, muprior, z, k, cov) {
  # Make output matrix
  out <- matrix(0L, nrow=ncol(x), ncol=k)
  # For each component, do
  for(i in 1:k) {
    prior <- muprior[[i]]
    # Smple size 
    sample.size = sum(z==i)
    # Get sample means
    if(sample.size == 0) {
      sample.means <- rep(0, k)
    } else {
      sample.means = apply(x[z==i,], 2, mean)
    }
    # Make inv. gamma matrix and the mean vectors
    inv_cov_prior <- solve(prior$cov)
    inv_cov <- solve(cov[[i]])
    step1 = solve(inv_cov_prior + sample.size * inv_cov)
    step2 = (inv_cov_prior %*% prior$mean) + (sample.size * inv_cov %*% sample.means)
    # Multiply
    mu_posterior <- step1 %*% step2
    # Sample 
    out[,i] <- mvtnorm::rmvnorm(1, mu_posterior, step1)
  }
  # Return
  return(out)
}

sample_mvn_cov <- function(x, sigmaprior, mu, z, k) {
  # Output matrix
  out <- vector("list", k)
  for(i in 1:k) {
    prior <- sigmaprior[[i]]
    sample.size = sum(z==i)
    # Compute nu1, psi1
    nu1 = sample.size + nu0
    xmoin = x[z==i,]
    xmoin[,1] <- xmoin[,1] - mu[1,i]
    xmoin[,2] <- xmoin[,2] - mu[2,i]
    psi1 = psi0 + t(xmoin) %*% (xmoin)
    out[[i]] <- MCMCpack::riwish(nu1, psi1)
  }
  # Return
  return(out)
}

#' @param x an n vector of data
#' @param pi a k vector
#' @param mu a k vector
sample_z = function(x,pi,mu, sigma, cov){
  dmat = matrix(0L, nrow=dim(x)[1], ncol=length(pi))
  dmat[,1] <- pi[1] * mvtnorm::dmvnorm(x, mean = mu[,1], cov[[1]])
  dmat[,2] <- pi[2] * mvtnorm::dmvnorm(x, mean = mu[,2], cov[[2]])
  #dmat = outer(mu,x,"-") # k by n matrix, d_kj =(mu_k - x_j)
  p.z.given.x = t(dmat)# as.vector(pi) * dnorm(t(dmat),mu,sigma) 
  p.z.given.x = apply(p.z.given.x,2,normalize) # normalize columns
  z = rep(0, dim(x)[1])
  for(i in 1:length(z)){
    z[i] = sample(1:length(pi), size=1,prob=p.z.given.x[,i],replace=TRUE)
  }
  return(z)
}

#' @param z an n vector of cluster allocations (1...k)
#' @param k the number of clusters
sample_pi = function(z,k){
  counts = colSums(outer(z,1:k,FUN="=="))
  pi = gtools::rdirichlet(1,counts+1)
  return(pi)
}

#' @param x an n vector of data
#' @param z an n vector of cluster allocations
#' @param k the number o clusters
#' @param prior.mean the prior mean for mu
#' @param prior.prec the prior precision for mu
sample_mu = function(x, z, k, prior, sigma){
  df = data.frame(x=x,z=z)
  mu = rep(0,k)
  for(i in 1:k){
    sample.size = sum(z==i)
    sample.mean = ifelse(sample.size==0,0,mean(x[z==i]))
    post.prec = (sample.size/sigma)+prior$prec
    post.mean = (prior$mean * prior$prec + (sample.mean * sample.size * 1/sigma))/post.prec
    mu[i] = rnorm(1,post.mean,sqrt(1/post.prec))
  }
  return(mu)
}

sample_sigma = function(x, z, k, prior, sigma, mu) {
  sigma = rep(0, k)
  for(i in 1:k) {
    sample.size = sum(z==i)
    shape = (sample.size/2) + prior$shape
    scale = sum((x[z==i]-mu[i])^2)/2 + prior$scale
    sigma[i] = 1 / rgamma(1, shape, scale)
  }
  return(sigma)
}

muprior <- list(
  list(
    "cov" = matrix(c(1000, -200, -200, 1000), nrow=2,ncol=2),
    "means" = c(0,0)
  ),
  list(
    "cov" = matrix(c(1000, -200, -200, 1000), nrow=2,ncol=2),
    "means" = c(0,0)
  )
)

sigmaprior <- list(
  list(
    nu=4, psi=matrix(c(1, .5, .5, 1), nrow=2, ncol=2)
  ),
  list(nu=4, psi=matrix(c(c(1, .5, .5, 1)), nrow=2, ncol=2))
)

niter =1000
gibbs = function(x,k,niter =1000,muprior = list(mean=0,prec=0.1), sigmaprior=list(nu=1, psi=matrix(rep(0.1, 2), nrow=2, ncol=2))){
  #pi = rep(1/k,k) # initialize
  #mu = rnorm(k,0,10)
  #sigma = c(1,1)
  cov = list(diag(k), diag(k))
  #mu = matrix(0L, nrow=ncol(x), ncol=k) + runif(k*ncol(x))
  # Sample latent
  #z = sample_z(x,pi,mu, sigma, cov)
  res = list(mu=matrix(nrow=niter, ncol=k*ncol(x)), pi = matrix(nrow=niter,ncol=k), z = matrix(nrow=niter, ncol=length(x)), cov=matrix(nrow=niter, ncol=k*k*k)) #  sigma = matrix(nrow=niter, ncol=k)
  res$mu[1,]=as.vector(mu)
  res$pi[1,]=pi
  res$z[1,]=z 
  res$cov[1,] = c(as.vector(cov[[1]]), as.vector(cov[[2]]))
  #res$sigma[1,] = sigma
  for(i in 2:niter){
    #pi = sample_pi(z,k)
    #mu = sample_mu(x,z,k,muprior, sigma)
    #mu = sample_mvn_mu(x, pi, muprior, z, k, cov)
    #sigma =  sqrt(sample_sigma(x,z,k,sigmaprior,sigma, mu))
    (cov = sample_mvn_cov(x, sigmaprior, mu, z, k))
    #z = sample_z(x,pi,mu, sigma, cov)
    res$mu[i,] = as.vector(mu)
    res$pi[i,] = pi
    res$z[i,] = z
    res$cov[i,] = c(as.vector(cov[[1]]), as.vector(cov[[2]]))
    #res$sigma[i,] = sigma
  }
  return(res)
}


burnout <- 1
#res = gibbs(x,2, muprior = muprior)
plot(res$mu[-1:-burnout,1],ylim=c(-4, 6),type="l")
lines(res$mu[-1:-burnout,2],col=2)

plot(res$mu[-1:-burnout,3],ylim=c(0,4),type="l")
lines(res$mu[-1:-burnout,4],col=2)

plot(res$pi[-1:-burnout,1], ylim=c(0,1), type="l")
lines(res$pi[-1:-burnout,2], col=2)

plot(res$cov[-1:-burnout,1], type="l")
plot(res$cov[-1:-burnout,2], type="l")
plot(res$cov[-1:-burnout,3], type="l")
plot(res$cov[-1:-burnout,4], type="l")

plot(res$cov[-1:-burnout,5], type="l")
plot(res$cov[-1:-burnout,6], type="l")
plot(res$cov[-1:-burnout,7], type="l")
plot(res$cov[-1:-burnout,8], type="l")

# MAP estimates
apply(res$mu[-1:-burnout,], 2, mean)

plot(res$sigma[,1], ylim=c(0,3), type="l")
lines(res$sigma[,2], col=2)
