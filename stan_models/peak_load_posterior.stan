/* Posterior model for peak load given sample measurement */
data {
    real low;
    real high;
    real error;
    real z;
}
parameters {
    real theta;
}
model {
    theta ~ uniform(low,high);
    z ~ normal(theta,error*theta);
}