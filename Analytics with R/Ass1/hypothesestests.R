cow.data = read.csv(file.choose())
cow.data$log_scc = log(cow.data$scc)


linearMod <- lm(log_scc ~protein, data=cow.data)

#Test if beta0 = 0 using T-test
#===========================
alpha = 0.05
alpha

N = length(cow.data$protein)
N

MSE = sum(linearMod$residuals^2/
            (length(cow.data$protein)-2))
MSE

SXX = sum((cow.data$protein-mean(cow.data$protein))^2)
SXX

VARB0 = MSE*(1/ length(cow.data$protein) 
             + (mean(cow.data$protein)^{2}/SXX)) 
VARB0

b0_T_ = (linearMod$coefficients[1]-0)/sqrt(VARB0) 
b0_T_

b0_TDIST = qt(1-0.05/2, N-2)
b0_TDIST

#Should we Reject
print("H0: b0 = 1 ; Ha: b0 != 1")
print(paste("We Should", str((if (abs(b0_T_)>b0_TDIST) "" else "not")), "Reject the hypotheses"))

b0_t_pval = 2 *( 1- pt(b0_T_, df = N - 2)) 
b0_t_pval
#===========================


#Test if beta1=0 using T-test
#===========================
alpha = 0.05
alpha

N = length(cow.data$protein)
N

MSE = sum(linearMod$residuals^2/(N-2))
MSE

SXX = sum((cow.data$protein-mean(cow.data$protein))^2)
SXX

VARB1 = MSE/SXX 
VARB1

T_b1 = (linearMod$coefficients[2]-0)/sqrt(VARB1) 
T_b1

TDIST_b1 = qt(1-0.05/2, N-2)
TDIST_b1

#Should we Reject
print("H0: b1 = 1 ; Ha: b1 != 1")
print(paste("We Should", str((if (abs(T_b1)>TDIST_b1) "" else "not")), "Reject the hypotheses"))

b1_t_pval = 2*(1-pt(T, df = N - 2)) 
b1_t_pval
#===========================

#Test if beta1=0 using F-test
#===========================
alpha = 0.05
N   = length(cow.data$protein)
MSR = sum((fitted(linearMod) - mean(cow.data$log_scc))^2) / 1
MSE = sum(linearMod$residuals^2)/(N-2)

b1_F = MSR/MSE 
b1_F

b1_FDIST = qf(1-alpha,1,N-2)
b1_FDIST

#Should we Reject
print("H0: b1 = 1 ; Ha: b1 != 1")
print(paste("We Should", str((if (abs(b1_F)>b1_FDIST) "" else "not")), "Reject the hypotheses"))

b1_f_pval = pf(1-b1_F, 1,N-2)
b1_f_pval
#===========================

summary(linearMod)

