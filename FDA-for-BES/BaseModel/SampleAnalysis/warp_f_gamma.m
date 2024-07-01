function fo = warp_f_gamma(f,gamma,t)
%
% Code from https://github.com/jdtuck/fdasrvf_MATLAB
%
fo = interp1(t, f, (t(end)-t(1)).*gamma + t(1))';