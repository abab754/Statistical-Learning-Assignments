## Instructions:
##
## Run the following code.
## You do not need to submit this file.
##

X = cbind(c(1,1,3,3,5,6,7), 11:17)

ixs = X[, 1] > 3

print(ixs)

X[ixs, ]
X[!ixs, ]

