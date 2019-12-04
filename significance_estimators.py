from numpy import sqrt, log, power

def z_asimov(s, b, sys=0.000001):

    if sys == 0 :
        return sqrt( 2.0*( (s+b)*log(1+s/b) -s ) )

    return sqrt( -2.0/(sys*sys)*log( b/( b+(b*b)*(sys*sys))*(sys*sys)*s+1.0)+ 2.0*(b+s)*log(( b+s)*( b+(b*b)*(sys*sys))/( (b*b)+( b+s)*(b*b)*(sys*sys))))

# def z_asimov(s, b, sys=0.000001):
#     return sqrt( -2.0*b/sys*log( sys*s/(b+b*sys)+1.0)+ 2.0*(s+b)*log( (s+b)*(b+b*sys)/( (b*b)+(s+b)*b*sys)))

def ez_asimov(s, es, b, eb, sys=0.000001):
    return power(-(eb*eb)/( 1.0/(sys*sys)*log( b/( b+(b*b)*(sys*sys))*(sys*sys)*s+1.0)-(
                b+s)*log(( b+s)*( b+(b*b)*(sys*sys))/( (b*b)+( b+s)*(b*b)*(sys*sys))))*power(
            1.0/( b/( b+(b*b)*(sys*sys))*(sys*sys)*s+1.0)/(sys*sys)*(1.0/( b+(b*b)*(sys*sys))*(
                    sys*sys)*s-b/power( b+(b*b)*(sys*sys),2.0)*(sys*sys)*( 2.0*b*(sys*sys)+1.0)*s)-(
                ( b+s)*( 2.0*b*(sys*sys)+1.0)/( (b*b)+( b+s)*(b*b)*(sys*sys))+( b+(b*b)*(sys*sys))/(
                    (b*b)+( b+s)*(b*b)*(sys*sys))-( b+s)*( 2.0*( b+s)*b*(sys*sys)+2.0*b+(b*b)*(sys*sys))*(
                    b+(b*b)*(sys*sys))/power( (b*b)+( b+s)*(b*b)*(sys*sys),2.0))/( b+(b*b)*(sys*sys))*(
                (b*b)+( b+s)*(b*b)*(sys*sys))-log(( b+s)*( b+(b*b)*(sys*sys))/(
                    (b*b)+( b+s)*(b*b)*(sys*sys))),2.0)/2.0-1.0/( 1.0/(sys*sys)*log(
                b/( b+(b*b)*(sys*sys))*(sys*sys)*s+1.0)-( b+s)*log(( b+s)*( b+(b*b)*(sys*sys))/(
                    (b*b)+( b+s)*(b*b)*(sys*sys))))*power( log(( b+s)*( b+(b*b)*(sys*sys))/(
                    (b*b)+( b+s)*(b*b)*(sys*sys)))+1.0/( b+(b*b)*(sys*sys))*(
                (b+(b*b)*(sys*sys))/( (b*b)+( b+s)*(b*b)*(sys*sys))-( b+s)*(b*b)*(
                    b+(b*b)*(sys*sys))*(sys*sys)/power( (b*b)+( b+s)*(b*b)*(sys*sys),2.0))*(
                (b*b)+( b+s)*(b*b)*(sys*sys))-1.0/( b/( b+(b*b)*(sys*sys))*(sys*sys)*s+1.0)*b/(
                b+(b*b)*(sys*sys)),2.0)*(es*es)/2.0,(1.0/2.0))

def asimov(sys = 0.1):
    return lambda s,b : z_asimov(s,b,sys)

def s_over_sqrt_of_b(s,b):
    return s/sqrt(b)


def z_asimov_with_reg(s, b, sys_rel=0.000001):  # sys_rel should be given as relative uncertainty -> sigma_b = sys_rel * b
    sigma_reg = 1.8410548  # this is  68% CL from the Neyman construction for N=0 -> thus the lowest statistical uncertainty that can be achieved (see : https://twiki.cern.ch/twiki/bin/viewauth/CMS/PoissonErrorBars)
    sigma_b = sqrt(sys_rel*sys_rel*b*b + sigma_reg*sigma_reg)
    return sqrt( 2.0*(s+b)*log((s+b)*( b+sigma_b*sigma_b)/( (b*b)+(s+b)*(sigma_b*sigma_b)))- 2.0*b*b/(sigma_b*sigma_b)*log( 1 + s*sigma_b*sigma_b/(b*(b+sigma_b*sigma_b))))


def asimov_with_reg(sys = 0.1):
    return lambda s,b : z_asimov_with_reg(s,b,sys)
    

def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )
    where b_r = 10, b = background, s = signal, log is natural logarithm """

    br = 10.0
    radicand = 2 *( (s+b+br) * log(1.0 + s/(b+br)) -s)

    return sqrt(radicand)
