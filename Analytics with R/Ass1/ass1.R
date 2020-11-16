#install.packages("ggpubr")
library("ggpubr")
library(psych)


cow.data = read.csv(file.choose())

#View sample of the data
head(cow.data)


pathf <- function(n){
  path_ = "Desktop/Sem 1/Predictive Analytics/Assignment1/graphs/"
  p <- paste(path_, n, sep="")
  return(p)
}


# Question 1
#===========================

png(file = pathf("sccboxplot.png"), 
    width     = 4,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
boxplot(cow.data$scc, col="lightgray", 
        main="Somatic Cell Count", ylab="SCC per Liter")
dev.off()
#Histogram Plot of SCC
png(file = pathf("scchist.png"), 
    width     = 4,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
hist(scc, xlab="SCC", breaks="fd", 
     main="Histogram of Distribution of Somatic Cell Count", col = "lightblue")
dev.off()
#Summary Stats of SCC
summary(cow.data$scc)
#===========================


# Question 2
#===========================
#log the SCC
lscc = log(scc)
#Box Plot of log SCC
png(file = pathf("logsccboxplot.png"), 
    width     = 4,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
boxplot(lscc, col="lightgray", main="Log SCC", 
        ylab="Log SCC / L")
dev.off()
#Histogram of log SCC
png(file = pathf("logscchist.png"), 
    width     = 4,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
hist(lscc, xlab="SCC", breaks="fd", 
     main="Histogram of Distribution of Somatic Cell Count", col = "lightblue") 
dev.off()
#Summary Stats of log SCC
summary(lscc)
#===========================


# Question 3
#===========================
pro = cow.data$protein
#Box Plot of protein
png(file = pathf("proteinboxplot.png"), 
    width     = 4,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
boxplot(pro, col="lightgray", main="Protein", ylab="Protein")
dev.off()
#Histogram of protein
png(file = pathf("proteinhist.png"), 
    width     = 4,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
hist(pro, xlab="Protein", breaks="fd", 
     main="Histogram of Distribution of Protein", col = "lightblue")
dev.off()

#Summary Stats of protein
summary(pro)
#===========================


# Question 4
#===========================
cas = cow.data$casein
#Box Plot of casein
png(file = pathf("caseinboxplot.png"), 
    width     = 4,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
boxplot(cas, col = "lightgray", main="Casein", ylab="Casein")
dev.off()
#Histogram of casein
png(file = pathf("caseinhist.png"), 
    width     = 4,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
hist(cas, xlab="Casein", breaks="fd", 
     main="Histogram of Distribution of Casein", col = "lightblue") 
dev.off()
#Summary Stats of casein
summary(cas)
#===========================

# Question 5
#===========================
cow.data$conc_fed = as.factor(cow.data$conc_fed)
cf = cow.data$conc_fed
#frequency:
cf_freq = table(cf)
cf_freq
png(file = pathf("concfedfreq.png"), 
    width     = 4,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
barplot(cf_freq, 
        main="Percentage Concentrated Feed Frequencies", 
        xlab="Percentage Concentrated Feed",ylab="Frequency", col = "lightgray")
dev.off()
summary(cf_freq)

#proportions:
cf_props = prop.table(table(cf))
cf_props
png(file = pathf("confedprop.png"), 
    width     = 4,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
barplot(cf_props, 
        main="Percentage Concentrated Feed Proportions", 
        xlab="Percentage Concentrated Feed",ylab="Proportion",col = "lightgray")
dev.off()
summary(cf_props)
#===========================


# Question 6
#===========================
cow.data$log_scc = lscc
png(file = pathf("logsccvarconcfed.png"), 
    width     = 4,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
boxplot(cow.data$log_scc~cow.data$conc_fed,
      main="Log SCC variation with Percentage Concentrated Feed",
      xlab="Percentage Concentrated Feed", ylab="Log Somatic Cell Count", 
      col="lightgray")
dev.off()
describeBy(cow.data$log_scc, cow.data$conc_fed)
#===========================


# Question 7
#===========================
pro_lscc_cor = cor(cow.data$protein, cow.data$log_scc)
pro_lscc_cor
png(file = pathf("provslscc.png"), 
    width     = 4,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
plot(cow.data$protein, cow.data$log_scc, main="Protein vs Log SCC",
     xlab = "Protein", ylab = "Log Somatic Cell Count", 
     type = "p", cex=1, pch=20, col=rgb(0, 0, 1, 0.8), lwd=0)
dev.off()
new_data = data.frame(cow.data$protein, cow.data$log_scc)

png(file = pathf("pairslsccpro.png"), 
    width     = 4,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
pairs(new_data, col="lightgray")
dev.off()

cas_lscc_cor = cor(cow.data$casein, cow.data$log_scc)
cas_lscc_cor
png(file = pathf("caseinvlscc.png"), 
    width     = 4,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
plot(cow.data$casein, cow.data$log_scc, main="Casein vs Log SCC",
     xlab = "Protein", ylab = "Log Somatic Cell Count", 
     type = "p", cex=1, pch=20, col=rgb(0, 0, 1, 0.8), lwd=0)
dev.off()
new_data = data.frame(cow.data$casein, cow.data$log_scc)
png(file = pathf("pairslscccaesin.png"), 
    width     = 4,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
pairs(new_data, col="lightgray")
dev.off()
#===========================

#Q8
#===========================
#Protein. higher correlation.
#===========================

#===========================
# REGRESSION MODEL
#===========================


#Q1
#===========================
png(file = pathf("proteinvslscc4.png"), 
    width     = 10,
    height    = 8,
    units     = "in",
    res       = 2400,
    pointsize = 5)
par(mfrow=c(2, 2))
boxplot(cow.data$protein, main="Protein", 
        par=par(family = 'Times New Roman', cex.axis=2, cex.lab=2, cex.main=2))
boxplot(cow.data$log_scc, main="Log Scc", 
        par = par(family = 'Times New Roman', cex.axis=2, cex.lab=2, cex.main=2))

x <- seq(-4, 4, length=100)
y <- dnorm(x)
par(family = 'Times New Roman', cex.axis=2, cex.lab=2, cex.main=2)
plot(density(cow.data$protein), main="Density Plot: Protein")
polygon(density(cow.data$protein), col="red")
par(family = 'Times New Roman', cex.axis=2, cex.lab=2, cex.main=2)
plot(density(cow.data$log_scc), main="Density Plot: Log SCC")
polygon(density(cow.data$log_scc), col="red")
dev.off()


population_mean <- mean(cow.data$protein)
population_sd <- sd(cow.data$protein)
lower_bound <- population_mean - population_sd
upper_bound <- population_mean + population_sd
x <- seq(-10, 10, length = 1000) * population_sd + population_mean
y <- dnorm(x, population_mean, population_sd)
png(file = pathf("proteindens_plot.png"), 
    width     = 5,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(mfrow=c(1,1))
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
plot(density(cow.data$protein), main="Density Plot: Protein", ylim=c(0, 1.5))
polygon(density(cow.data$protein), col=rgb(0, 1, 0, 0.4))
lines(x, y, col="red", lwd=1)
polygon(x,y, col=rgb(1, 0, 0, 0.1), border = NA)
dev.off()

population_mean <- mean(cow.data$log_scc)
population_sd <- sd(cow.data$log_scc)
lower_bound <- population_mean - population_sd
upper_bound <- population_mean + population_sd
x <- seq(-4, 4, length = 1000) * population_sd + population_mean
y <- dnorm(x, population_mean, population_sd)

png(file = pathf("lsccndens_plot.png"), 
    width     = 5,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(mfrow=c(1,1))
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
plot(density(cow.data$log_scc), main="Density Plot: Log SCC")
polygon(density(cow.data$log_scc), col=rgb(0, 1, 0, 0.4))
lines(x, y, col="red", lwd=1)
polygon(x,y, col=rgb(1, 0, 0, 0.1), border = NA)
dev.off()


linearMod <- lm(log_scc ~protein, data=cow.data)

linearMod$coefficients

png(file = pathf("fitted_line.png"), 
    width     = 5,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(mfrow=c(1,1))
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
plot(cow.data$protein, cow.data$log_scc, main="Fitted Line",
     xlab = "Protein", ylab = "Somatic Cell Count (Log)", 
     col="blue", type = "p", cex=0.5, pch=10,)
lines(cow.data$protein, fitted(linearMod), col="red", lwd=2)
dev.off()

b0 = as.numeric(linearMod$coefficients[1:1])
b0
b1 = as.numeric(linearMod$coefficients[2:2])
b1
#===========================

#Question 2
#===========================
# Intercept is 2.3462 which means when the protein level is 0 
# we would expect the Log of the Somatic Cell count to be 2.3462
#===========================

#Question 3
#===========================
# The slope is 0.3955 which means for every 1 step increase in protein there is 
# a 0.39 step increase in Log somatic Cell Count
#===========================



#Question 4
#===========================
N   = length(cow.data$protein)
MSE = sum(linearMod$residuals^2/(N-2))
SXX = sum((cow.data$protein-mean(cow.data$protein))^2)
VARB0 = MSE*(1/ N + (mean(cow.data$protein)^{2}/SXX))
VARB0

N   = length(cow.data$protein)
SSE  = sum((cow.data$log_scc - linearMod$fitted.values)^2)
MSE  = SSE/(N-2)
SXX  = sum((cow.data$protein - mean(cow.data$protein))^2)
VARB1 = MSE/SXX
VARB1
#===========================

#Question 5
#===========================
N   = length(cow.data$protein)
MSE = sum(linearMod$residuals^2/(N-2))
SXX = sum((cow.data$protein-mean(cow.data$protein))^2)
VARB0 = MSE*(1/ N + (mean(cow.data$protein)^{2}/SXX)) 

alpha=0.05
beta0 = linearMod$coefficients[1]
c(beta0 - qt(1-alpha/2,N-2)*sqrt(VARB0),
  beta0 + qt(1-alpha/2,N-2)*sqrt(VARB0))

confint(linearMod)
#===========================

#Question 6
#===========================
N   = length(cow.data$protein)
SSE  = sum((cow.data$log_scc - linearMod$fitted.values)^2)
MSE  = SSE/(N-2)
SXX  = sum((cow.data$protein - mean(cow.data$protein))^2)

VARB1 = MSE/SXX

beta1= linearMod$coefficients[2]
alpha=0.05
c(beta1 - qt(1-alpha/2,N-2)*sqrt(VARB1),
  beta1 + qt(1-alpha/2,N-2)*sqrt(VARB1))
confint(linearMod)
#===========================


# Quetstion 7
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

summary(linearMod)

#Should we Reject
print("H0: b0 = 1 ; Ha: b0 != 1")
print(paste("We Should", str((if (abs(b0_T_)>b0_TDIST) "" else "not")), "Reject the hypotheses"))

b0_t_pval = 2 *( 1- pt(b0_T_, df = N - 2)) 
b0_t_pval
#===========================


# Quetstion 8
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

b1_t_pval = 2*(1-pt(T_b1, df = N - 2)) 
b1_t_pval


summary(linearMod)
#Test if beta1=0 using F-test
alpha = 0.05
N   = length(cow.data$protein)
MSR = sum((fitted(linearMod) - mean(cow.data$log_scc))^2) / 1
MSR

MSE = sum(linearMod$residuals^2)/(N-2)
MSE

b1_F = MSR/MSE 
b1_F

b1_FDIST = qf(1-alpha,1,N-2)
b1_FDIST

#Should we Reject
print("H0: b1 = 1 ; Ha: b1 != 1")
print(paste("We Should", str((if (abs(b1_F)>b1_FDIST) "" else "not")), "Reject the hypotheses"))

b1_f_pval = pf(b1_F, 1,N-2)
b1_f_pval
#===========================


#Question 9
#===========================
summary_model = summary(linearMod)
summary_model
summary_model$fstatistic
f <- summary(linearMod)$fstatistic  # parameters for model p-value calc
model_p <- pf(f[1], f[2], f[3], lower=FALSE)
model_p
#===========================

#Question 10
#===========================
#Calculate the coefficient of determination
SYY = sum((cow.data$log_scc-mean(cow.data$log_scc))^2)
SSE = sum(linearMod$residuals^2)
R2 <- (SYY - SSE) /SYY
R2

summary(linearMod)$r.squared
#===========================


#Question 11
#===========================
RMSE = sqrt(SSE/(length(cow.data$log_scc)-2))
RMSE

summary_model
#===========================


#Question 12
#===========================
alpha = 0.05

N = length(cow.data$protein)
N

SXX = sum((cow.data$protein - mean(cow.data$protein))^2)
SXX

MSE = sum(linearMod$residuals^2/(N-2))
MSE

VAR_Y = MSE*(1/N+(cow.data$protein-mean(cow.data$protein))^2/SXX)
VAR_Y

Yhat = fitted(linearMod)
Yhat

YCI_HI = Yhat + qt(1-alpha/2,N-2)*sqrt(VAR_Y)
YCI_HI

YCI_LO = Yhat - qt(1-alpha/2,N-2)*sqrt(VAR_Y)
YCI_LO

png(file = pathf("errorbars.png"), 
    width     = 5,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(mfrow=c(1,1))
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
plot(cow.data$protein,cow.data$log_scc,
     xlab="Protein", ylab="Log Somatic Cell Count", 
     col="black", type = "p", cex=0.5, pch=1, lwd=0.5)
abline(linearMod, col="blue", lwd=2)

df1 = data.frame(cbind(cow.data$protein,YCI_HI))
orderidx = order(df1[, "V1"])
df1 = df1[orderidx, ,drop=FALSE]
lines(df1$V1,df1$YCI_HI,col="red", type = "l", lwd=1 )

df2 = data.frame(cbind(cow.data$protein,YCI_LO))
  orderidx = order(df2[, "V1"])
df2 = df2[orderidx, ,drop=FALSE]
lines(df2$V1,df2$YCI_LO,col="red", type = "l", lwd=1 )

polygon(c(df1$V1, rev(df2$V1)), c(df1$YCI_HI, rev(df2$YCI_LO)), col=rgb(1, 0, 0, 0.2), border = NA)
dev.off()
#===========================

#Question 13 Assumptions
#===========================
# Variation in X
# Random Sampling
# Linearity in Parameters
# Zero Conditional Mean
# Homoskedacity
# Normality of Errors
#===========================


#Question 14 
#===========================

# Variation in X
xvar = var(cow.data$protein)
xvar

SXX = sum((cow.data$protein-mean(cow.data$protein))^2)
SXX

# Random Sampling
# Plot of protein doesn't have distinct features

#Check if we have zero conditional mean and constant variance
summary(residuals(linearMod))
dev.off()
png(file = pathf("proteinvsresiduals.png"), 
    width     = 5,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(mfrow=c(1,1))
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
plot(cow.data$protein,residuals(linearMod), main= "Protein vs Residuals",col="blue")
dev.off()

# Linearity in Parameters
# scatter plot shows a "generally linear relationship.

# Homoskedacity
# check fitted line. whats the variation about it?

# Normality of Errors
png(file = pathf("normalityoferrors.png"), 
    width     = 5,
    height    = 3,
    units     = "in",
    res       = 1200,
    pointsize = 5)
par(mfrow=c(1,1))
par(family = 'Times New Roman', cex.axis=1.5, cex.lab=1.5, cex.main=2)
plot(density(residuals(linearMod)),main="Density Plot: Residuals")
polygon(density(residuals(linearMod)))
polygon(density(residuals(linearMod)), col=rgb(0, 1, 0, 0.4))
population_mean <- mean(residuals(linearMod))
population_sd <- sd(residuals(linearMod))
lower_bound <- population_mean - population_sd
upper_bound <- population_mean + population_sd
x <- seq(-4, 4, length = 1000) * population_sd + population_mean
y <- dnorm(x, population_mean, population_sd)
lines(x, y, col="red", lwd=1)
polygon(x,y, col=rgb(1, 0, 0, 0.1), border = NA)
dev.off()

shapiro.test(residuals(linearMod))
#===========================



