from numpy import sqrt, log, power

def z_asimov(s, b, sys=0.000001):
    return sqrt( -2.0/(sys*sys)*log( b/( b+(b*b)*(sys*sys))*(sys*sys)*s+1.0)+ 2.0*(b+s)*log(( b+s)*( b+(b*b)*(sys*sys))/( (b*b)+( b+s)*(b*b)*(sys*sys))))

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



    
