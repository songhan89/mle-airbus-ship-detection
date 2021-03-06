??(
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8ָ
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_19/kernel
}
$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*&
_output_shapes
:*
dtype0
t
conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_19/bias
m
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
_output_shapes
:*
dtype0
?
conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_20/kernel
}
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*&
_output_shapes
:*
dtype0
t
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_20/bias
m
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes
:*
dtype0
?
conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_21/kernel
}
$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*&
_output_shapes
:*
dtype0
t
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_21/bias
m
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes
:*
dtype0
?
conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_22/kernel
}
$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*&
_output_shapes
:*
dtype0
t
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_22/bias
m
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes
:*
dtype0
?
conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_23/kernel
}
$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*&
_output_shapes
: *
dtype0
t
conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_23/bias
m
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
_output_shapes
: *
dtype0
?
conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_24/kernel
}
$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_24/bias
m
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes
: *
dtype0
?
conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_25/kernel
}
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_25/bias
m
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
_output_shapes
:@*
dtype0
?
conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_26/kernel
}
$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_26/bias
m
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
_output_shapes
:@*
dtype0
?
conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*!
shared_nameconv2d_27/kernel
~
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*'
_output_shapes
:@?*
dtype0
u
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_27/bias
n
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes	
:?*
dtype0
?
conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_28/kernel

$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_28/bias
n
"conv2d_28/bias/Read/ReadVariableOpReadVariableOpconv2d_28/bias*
_output_shapes	
:?*
dtype0
?
conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*!
shared_nameconv2d_29/kernel
~
$conv2d_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_29/kernel*'
_output_shapes
:?@*
dtype0
t
conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_29/bias
m
"conv2d_29/bias/Read/ReadVariableOpReadVariableOpconv2d_29/bias*
_output_shapes
:@*
dtype0
?
conv2d_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_30/kernel
}
$conv2d_30/kernel/Read/ReadVariableOpReadVariableOpconv2d_30/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_30/bias
m
"conv2d_30/bias/Read/ReadVariableOpReadVariableOpconv2d_30/bias*
_output_shapes
:@*
dtype0
?
conv2d_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:` *!
shared_nameconv2d_31/kernel
}
$conv2d_31/kernel/Read/ReadVariableOpReadVariableOpconv2d_31/kernel*&
_output_shapes
:` *
dtype0
t
conv2d_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_31/bias
m
"conv2d_31/bias/Read/ReadVariableOpReadVariableOpconv2d_31/bias*
_output_shapes
: *
dtype0
?
conv2d_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_32/kernel
}
$conv2d_32/kernel/Read/ReadVariableOpReadVariableOpconv2d_32/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_32/bias
m
"conv2d_32/bias/Read/ReadVariableOpReadVariableOpconv2d_32/bias*
_output_shapes
: *
dtype0
?
conv2d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*!
shared_nameconv2d_33/kernel
}
$conv2d_33/kernel/Read/ReadVariableOpReadVariableOpconv2d_33/kernel*&
_output_shapes
:0*
dtype0
t
conv2d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_33/bias
m
"conv2d_33/bias/Read/ReadVariableOpReadVariableOpconv2d_33/bias*
_output_shapes
:*
dtype0
?
conv2d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_34/kernel
}
$conv2d_34/kernel/Read/ReadVariableOpReadVariableOpconv2d_34/kernel*&
_output_shapes
:*
dtype0
t
conv2d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_34/bias
m
"conv2d_34/bias/Read/ReadVariableOpReadVariableOpconv2d_34/bias*
_output_shapes
:*
dtype0
?
conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_35/kernel
}
$conv2d_35/kernel/Read/ReadVariableOpReadVariableOpconv2d_35/kernel*&
_output_shapes
:*
dtype0
t
conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_35/bias
m
"conv2d_35/bias/Read/ReadVariableOpReadVariableOpconv2d_35/bias*
_output_shapes
:*
dtype0
?
conv2d_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_36/kernel
}
$conv2d_36/kernel/Read/ReadVariableOpReadVariableOpconv2d_36/kernel*&
_output_shapes
:*
dtype0
t
conv2d_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_36/bias
m
"conv2d_36/bias/Read/ReadVariableOpReadVariableOpconv2d_36/bias*
_output_shapes
:*
dtype0
?
conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_37/kernel
}
$conv2d_37/kernel/Read/ReadVariableOpReadVariableOpconv2d_37/kernel*&
_output_shapes
:*
dtype0
t
conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_37/bias
m
"conv2d_37/bias/Read/ReadVariableOpReadVariableOpconv2d_37/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_1/gamma/m
?
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:*
dtype0
?
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_1/beta/m
?
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_19/kernel/m
?
+Adam/conv2d_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_19/bias/m
{
)Adam/conv2d_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_20/kernel/m
?
+Adam/conv2d_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_20/bias/m
{
)Adam/conv2d_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_21/kernel/m
?
+Adam/conv2d_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_21/bias/m
{
)Adam/conv2d_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_22/kernel/m
?
+Adam/conv2d_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_22/bias/m
{
)Adam/conv2d_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_23/kernel/m
?
+Adam/conv2d_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_23/bias/m
{
)Adam/conv2d_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_24/kernel/m
?
+Adam/conv2d_24/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/kernel/m*&
_output_shapes
:  *
dtype0
?
Adam/conv2d_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_24/bias/m
{
)Adam/conv2d_24/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_25/kernel/m
?
+Adam/conv2d_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_25/bias/m
{
)Adam/conv2d_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_26/kernel/m
?
+Adam/conv2d_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_26/bias/m
{
)Adam/conv2d_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_27/kernel/m
?
+Adam/conv2d_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_27/bias/m
|
)Adam/conv2d_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_28/kernel/m
?
+Adam/conv2d_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_28/bias/m
|
)Adam/conv2d_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*(
shared_nameAdam/conv2d_29/kernel/m
?
+Adam/conv2d_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/kernel/m*'
_output_shapes
:?@*
dtype0
?
Adam/conv2d_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_29/bias/m
{
)Adam/conv2d_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_30/kernel/m
?
+Adam/conv2d_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_30/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_30/bias/m
{
)Adam/conv2d_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_30/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:` *(
shared_nameAdam/conv2d_31/kernel/m
?
+Adam/conv2d_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/kernel/m*&
_output_shapes
:` *
dtype0
?
Adam/conv2d_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_31/bias/m
{
)Adam/conv2d_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_32/kernel/m
?
+Adam/conv2d_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/kernel/m*&
_output_shapes
:  *
dtype0
?
Adam/conv2d_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_32/bias/m
{
)Adam/conv2d_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*(
shared_nameAdam/conv2d_33/kernel/m
?
+Adam/conv2d_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/kernel/m*&
_output_shapes
:0*
dtype0
?
Adam/conv2d_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_33/bias/m
{
)Adam/conv2d_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_34/kernel/m
?
+Adam/conv2d_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_34/bias/m
{
)Adam/conv2d_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_35/kernel/m
?
+Adam/conv2d_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_35/bias/m
{
)Adam/conv2d_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_36/kernel/m
?
+Adam/conv2d_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_36/bias/m
{
)Adam/conv2d_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_37/kernel/m
?
+Adam/conv2d_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_37/bias/m
{
)Adam/conv2d_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/bias/m*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_1/gamma/v
?
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:*
dtype0
?
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_1/beta/v
?
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_19/kernel/v
?
+Adam/conv2d_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_19/bias/v
{
)Adam/conv2d_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_20/kernel/v
?
+Adam/conv2d_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_20/bias/v
{
)Adam/conv2d_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_21/kernel/v
?
+Adam/conv2d_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_21/bias/v
{
)Adam/conv2d_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_22/kernel/v
?
+Adam/conv2d_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_22/bias/v
{
)Adam/conv2d_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_23/kernel/v
?
+Adam/conv2d_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_23/bias/v
{
)Adam/conv2d_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_24/kernel/v
?
+Adam/conv2d_24/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/kernel/v*&
_output_shapes
:  *
dtype0
?
Adam/conv2d_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_24/bias/v
{
)Adam/conv2d_24/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_24/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_25/kernel/v
?
+Adam/conv2d_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_25/bias/v
{
)Adam/conv2d_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_26/kernel/v
?
+Adam/conv2d_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_26/bias/v
{
)Adam/conv2d_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_27/kernel/v
?
+Adam/conv2d_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/conv2d_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_27/bias/v
|
)Adam/conv2d_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_27/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_28/kernel/v
?
+Adam/conv2d_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_28/bias/v
|
)Adam/conv2d_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_28/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*(
shared_nameAdam/conv2d_29/kernel/v
?
+Adam/conv2d_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/kernel/v*'
_output_shapes
:?@*
dtype0
?
Adam/conv2d_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_29/bias/v
{
)Adam/conv2d_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_29/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_30/kernel/v
?
+Adam/conv2d_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_30/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_30/bias/v
{
)Adam/conv2d_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_30/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:` *(
shared_nameAdam/conv2d_31/kernel/v
?
+Adam/conv2d_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/kernel/v*&
_output_shapes
:` *
dtype0
?
Adam/conv2d_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_31/bias/v
{
)Adam/conv2d_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_32/kernel/v
?
+Adam/conv2d_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/kernel/v*&
_output_shapes
:  *
dtype0
?
Adam/conv2d_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_32/bias/v
{
)Adam/conv2d_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_32/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*(
shared_nameAdam/conv2d_33/kernel/v
?
+Adam/conv2d_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/kernel/v*&
_output_shapes
:0*
dtype0
?
Adam/conv2d_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_33/bias/v
{
)Adam/conv2d_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_33/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_34/kernel/v
?
+Adam/conv2d_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_34/bias/v
{
)Adam/conv2d_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_35/kernel/v
?
+Adam/conv2d_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_35/bias/v
{
)Adam/conv2d_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_36/kernel/v
?
+Adam/conv2d_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_36/bias/v
{
)Adam/conv2d_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_37/kernel/v
?
+Adam/conv2d_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_37/bias/v
{
)Adam/conv2d_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer-17
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer-21
layer-22
layer_with_weights-13
layer-23
layer_with_weights-14
layer-24
layer-25
layer-26
layer_with_weights-15
layer-27
layer_with_weights-16
layer-28
layer-29
layer-30
 layer_with_weights-17
 layer-31
!layer_with_weights-18
!layer-32
"layer_with_weights-19
"layer-33
#layer-34
$layer-35
%	optimizer
&regularization_losses
'	variables
(trainable_variables
)	keras_api
*
signatures
 
R
+	variables
,regularization_losses
-trainable_variables
.	keras_api
?
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5regularization_losses
6trainable_variables
7	keras_api
h

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
h

>kernel
?bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
R
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
h

Hkernel
Ibias
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
h

Nkernel
Obias
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
R
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
h

Xkernel
Ybias
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
h

^kernel
_bias
`	variables
aregularization_losses
btrainable_variables
c	keras_api
R
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
h

hkernel
ibias
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
h

nkernel
obias
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
R
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
h

xkernel
ybias
z	variables
{regularization_losses
|trainable_variables
}	keras_api
l

~kernel
bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
n
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate0m?1m?8m?9m?>m??m?Hm?Im?Nm?Om?Xm?Ym?^m?_m?hm?im?nm?om?xm?ym?~m?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?0v?1v?8v?9v?>v??v?Hv?Iv?Nv?Ov?Xv?Yv?^v?_v?hv?iv?nv?ov?xv?yv?~v?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?
 
?
00
11
22
33
84
95
>6
?7
H8
I9
N10
O11
X12
Y13
^14
_15
h16
i17
n18
o19
x20
y21
~22
23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?
00
11
82
93
>4
?5
H6
I7
N8
O9
X10
Y11
^12
_13
h14
i15
n16
o17
x18
y19
~20
21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?
?metrics
?layer_metrics
?layers
 ?layer_regularization_losses
&regularization_losses
'	variables
(trainable_variables
?non_trainable_variables
 
 
 
 
?
?metrics
+	variables
?layer_metrics
?layers
 ?layer_regularization_losses
,regularization_losses
-trainable_variables
?non_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

00
11
22
33
 

00
11
?
?metrics
4	variables
?layer_metrics
?layers
 ?layer_regularization_losses
5regularization_losses
6trainable_variables
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_19/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_19/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91
 

80
91
?
?metrics
:	variables
?layer_metrics
?layers
 ?layer_regularization_losses
;regularization_losses
<trainable_variables
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_20/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_20/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

>0
?1
 

>0
?1
?
?metrics
@	variables
?layer_metrics
?layers
 ?layer_regularization_losses
Aregularization_losses
Btrainable_variables
?non_trainable_variables
 
 
 
?
?metrics
D	variables
?layer_metrics
?layers
 ?layer_regularization_losses
Eregularization_losses
Ftrainable_variables
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_21/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_21/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

H0
I1
 

H0
I1
?
?metrics
J	variables
?layer_metrics
?layers
 ?layer_regularization_losses
Kregularization_losses
Ltrainable_variables
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_22/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_22/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

N0
O1
 

N0
O1
?
?metrics
P	variables
?layer_metrics
?layers
 ?layer_regularization_losses
Qregularization_losses
Rtrainable_variables
?non_trainable_variables
 
 
 
?
?metrics
T	variables
?layer_metrics
?layers
 ?layer_regularization_losses
Uregularization_losses
Vtrainable_variables
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_23/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_23/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

X0
Y1
 

X0
Y1
?
?metrics
Z	variables
?layer_metrics
?layers
 ?layer_regularization_losses
[regularization_losses
\trainable_variables
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_24/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_24/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

^0
_1
 

^0
_1
?
?metrics
`	variables
?layer_metrics
?layers
 ?layer_regularization_losses
aregularization_losses
btrainable_variables
?non_trainable_variables
 
 
 
?
?metrics
d	variables
?layer_metrics
?layers
 ?layer_regularization_losses
eregularization_losses
ftrainable_variables
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_25/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_25/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

h0
i1
 

h0
i1
?
?metrics
j	variables
?layer_metrics
?layers
 ?layer_regularization_losses
kregularization_losses
ltrainable_variables
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_26/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_26/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

n0
o1
 

n0
o1
?
?metrics
p	variables
?layer_metrics
?layers
 ?layer_regularization_losses
qregularization_losses
rtrainable_variables
?non_trainable_variables
 
 
 
?
?metrics
t	variables
?layer_metrics
?layers
 ?layer_regularization_losses
uregularization_losses
vtrainable_variables
?non_trainable_variables
\Z
VARIABLE_VALUEconv2d_27/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_27/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

x0
y1
 

x0
y1
?
?metrics
z	variables
?layer_metrics
?layers
 ?layer_regularization_losses
{regularization_losses
|trainable_variables
?non_trainable_variables
][
VARIABLE_VALUEconv2d_28/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_28/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

~0
1
 

~0
1
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
 
 
 
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
 
 
 
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
][
VARIABLE_VALUEconv2d_29/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_29/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
][
VARIABLE_VALUEconv2d_30/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_30/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
 
 
 
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
 
 
 
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
][
VARIABLE_VALUEconv2d_31/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_31/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
][
VARIABLE_VALUEconv2d_32/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_32/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
 
 
 
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
 
 
 
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
][
VARIABLE_VALUEconv2d_33/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_33/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
][
VARIABLE_VALUEconv2d_34/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_34/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
 
 
 
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
 
 
 
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
][
VARIABLE_VALUEconv2d_35/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_35/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
][
VARIABLE_VALUEconv2d_36/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_36/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
][
VARIABLE_VALUEconv2d_37/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_37/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1
 

?0
?1
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
 
 
 
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
 
 
 
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
?0
?1
?2
?3
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
 

20
31
 
 
 
 
 
 
 
 
 

20
31
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_19/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_19/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_20/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_20/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_21/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_21/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_22/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_22/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_23/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_23/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_24/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_24/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_25/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_25/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_26/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_26/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_27/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_27/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_28/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_28/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_29/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_29/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_30/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_30/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_31/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_31/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_32/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_32/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_33/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_33/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_34/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_34/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_35/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_35/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_36/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_36/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_37/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_37/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_19/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_19/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_20/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_20/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_21/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_21/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_22/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_22/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_23/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_23/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_24/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_24/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_25/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_25/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_26/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_26/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_27/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_27/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_28/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_28/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_29/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_29/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_30/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_30/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_31/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_31/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_32/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_32/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_33/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_33/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_34/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_34/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_35/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_35/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_36/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_36/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_37/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_37/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_RGB_InputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_RGB_Inputbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasconv2d_27/kernelconv2d_27/biasconv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/biasconv2d_30/kernelconv2d_30/biasconv2d_31/kernelconv2d_31/biasconv2d_32/kernelconv2d_32/biasconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/biasconv2d_35/kernelconv2d_35/biasconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/bias*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_37261
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?.
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp$conv2d_21/kernel/Read/ReadVariableOp"conv2d_21/bias/Read/ReadVariableOp$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp$conv2d_23/kernel/Read/ReadVariableOp"conv2d_23/bias/Read/ReadVariableOp$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp$conv2d_25/kernel/Read/ReadVariableOp"conv2d_25/bias/Read/ReadVariableOp$conv2d_26/kernel/Read/ReadVariableOp"conv2d_26/bias/Read/ReadVariableOp$conv2d_27/kernel/Read/ReadVariableOp"conv2d_27/bias/Read/ReadVariableOp$conv2d_28/kernel/Read/ReadVariableOp"conv2d_28/bias/Read/ReadVariableOp$conv2d_29/kernel/Read/ReadVariableOp"conv2d_29/bias/Read/ReadVariableOp$conv2d_30/kernel/Read/ReadVariableOp"conv2d_30/bias/Read/ReadVariableOp$conv2d_31/kernel/Read/ReadVariableOp"conv2d_31/bias/Read/ReadVariableOp$conv2d_32/kernel/Read/ReadVariableOp"conv2d_32/bias/Read/ReadVariableOp$conv2d_33/kernel/Read/ReadVariableOp"conv2d_33/bias/Read/ReadVariableOp$conv2d_34/kernel/Read/ReadVariableOp"conv2d_34/bias/Read/ReadVariableOp$conv2d_35/kernel/Read/ReadVariableOp"conv2d_35/bias/Read/ReadVariableOp$conv2d_36/kernel/Read/ReadVariableOp"conv2d_36/bias/Read/ReadVariableOp$conv2d_37/kernel/Read/ReadVariableOp"conv2d_37/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp+Adam/conv2d_19/kernel/m/Read/ReadVariableOp)Adam/conv2d_19/bias/m/Read/ReadVariableOp+Adam/conv2d_20/kernel/m/Read/ReadVariableOp)Adam/conv2d_20/bias/m/Read/ReadVariableOp+Adam/conv2d_21/kernel/m/Read/ReadVariableOp)Adam/conv2d_21/bias/m/Read/ReadVariableOp+Adam/conv2d_22/kernel/m/Read/ReadVariableOp)Adam/conv2d_22/bias/m/Read/ReadVariableOp+Adam/conv2d_23/kernel/m/Read/ReadVariableOp)Adam/conv2d_23/bias/m/Read/ReadVariableOp+Adam/conv2d_24/kernel/m/Read/ReadVariableOp)Adam/conv2d_24/bias/m/Read/ReadVariableOp+Adam/conv2d_25/kernel/m/Read/ReadVariableOp)Adam/conv2d_25/bias/m/Read/ReadVariableOp+Adam/conv2d_26/kernel/m/Read/ReadVariableOp)Adam/conv2d_26/bias/m/Read/ReadVariableOp+Adam/conv2d_27/kernel/m/Read/ReadVariableOp)Adam/conv2d_27/bias/m/Read/ReadVariableOp+Adam/conv2d_28/kernel/m/Read/ReadVariableOp)Adam/conv2d_28/bias/m/Read/ReadVariableOp+Adam/conv2d_29/kernel/m/Read/ReadVariableOp)Adam/conv2d_29/bias/m/Read/ReadVariableOp+Adam/conv2d_30/kernel/m/Read/ReadVariableOp)Adam/conv2d_30/bias/m/Read/ReadVariableOp+Adam/conv2d_31/kernel/m/Read/ReadVariableOp)Adam/conv2d_31/bias/m/Read/ReadVariableOp+Adam/conv2d_32/kernel/m/Read/ReadVariableOp)Adam/conv2d_32/bias/m/Read/ReadVariableOp+Adam/conv2d_33/kernel/m/Read/ReadVariableOp)Adam/conv2d_33/bias/m/Read/ReadVariableOp+Adam/conv2d_34/kernel/m/Read/ReadVariableOp)Adam/conv2d_34/bias/m/Read/ReadVariableOp+Adam/conv2d_35/kernel/m/Read/ReadVariableOp)Adam/conv2d_35/bias/m/Read/ReadVariableOp+Adam/conv2d_36/kernel/m/Read/ReadVariableOp)Adam/conv2d_36/bias/m/Read/ReadVariableOp+Adam/conv2d_37/kernel/m/Read/ReadVariableOp)Adam/conv2d_37/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp+Adam/conv2d_19/kernel/v/Read/ReadVariableOp)Adam/conv2d_19/bias/v/Read/ReadVariableOp+Adam/conv2d_20/kernel/v/Read/ReadVariableOp)Adam/conv2d_20/bias/v/Read/ReadVariableOp+Adam/conv2d_21/kernel/v/Read/ReadVariableOp)Adam/conv2d_21/bias/v/Read/ReadVariableOp+Adam/conv2d_22/kernel/v/Read/ReadVariableOp)Adam/conv2d_22/bias/v/Read/ReadVariableOp+Adam/conv2d_23/kernel/v/Read/ReadVariableOp)Adam/conv2d_23/bias/v/Read/ReadVariableOp+Adam/conv2d_24/kernel/v/Read/ReadVariableOp)Adam/conv2d_24/bias/v/Read/ReadVariableOp+Adam/conv2d_25/kernel/v/Read/ReadVariableOp)Adam/conv2d_25/bias/v/Read/ReadVariableOp+Adam/conv2d_26/kernel/v/Read/ReadVariableOp)Adam/conv2d_26/bias/v/Read/ReadVariableOp+Adam/conv2d_27/kernel/v/Read/ReadVariableOp)Adam/conv2d_27/bias/v/Read/ReadVariableOp+Adam/conv2d_28/kernel/v/Read/ReadVariableOp)Adam/conv2d_28/bias/v/Read/ReadVariableOp+Adam/conv2d_29/kernel/v/Read/ReadVariableOp)Adam/conv2d_29/bias/v/Read/ReadVariableOp+Adam/conv2d_30/kernel/v/Read/ReadVariableOp)Adam/conv2d_30/bias/v/Read/ReadVariableOp+Adam/conv2d_31/kernel/v/Read/ReadVariableOp)Adam/conv2d_31/bias/v/Read/ReadVariableOp+Adam/conv2d_32/kernel/v/Read/ReadVariableOp)Adam/conv2d_32/bias/v/Read/ReadVariableOp+Adam/conv2d_33/kernel/v/Read/ReadVariableOp)Adam/conv2d_33/bias/v/Read/ReadVariableOp+Adam/conv2d_34/kernel/v/Read/ReadVariableOp)Adam/conv2d_34/bias/v/Read/ReadVariableOp+Adam/conv2d_35/kernel/v/Read/ReadVariableOp)Adam/conv2d_35/bias/v/Read/ReadVariableOp+Adam/conv2d_36/kernel/v/Read/ReadVariableOp)Adam/conv2d_36/bias/v/Read/ReadVariableOp+Adam/conv2d_37/kernel/v/Read/ReadVariableOp)Adam/conv2d_37/bias/v/Read/ReadVariableOpConst*?
Tin?
?2?	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_38863
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasconv2d_27/kernelconv2d_27/biasconv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/biasconv2d_30/kernelconv2d_30/biasconv2d_31/kernelconv2d_31/biasconv2d_32/kernelconv2d_32/biasconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/biasconv2d_35/kernelconv2d_35/biasconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2total_3count_3"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/conv2d_19/kernel/mAdam/conv2d_19/bias/mAdam/conv2d_20/kernel/mAdam/conv2d_20/bias/mAdam/conv2d_21/kernel/mAdam/conv2d_21/bias/mAdam/conv2d_22/kernel/mAdam/conv2d_22/bias/mAdam/conv2d_23/kernel/mAdam/conv2d_23/bias/mAdam/conv2d_24/kernel/mAdam/conv2d_24/bias/mAdam/conv2d_25/kernel/mAdam/conv2d_25/bias/mAdam/conv2d_26/kernel/mAdam/conv2d_26/bias/mAdam/conv2d_27/kernel/mAdam/conv2d_27/bias/mAdam/conv2d_28/kernel/mAdam/conv2d_28/bias/mAdam/conv2d_29/kernel/mAdam/conv2d_29/bias/mAdam/conv2d_30/kernel/mAdam/conv2d_30/bias/mAdam/conv2d_31/kernel/mAdam/conv2d_31/bias/mAdam/conv2d_32/kernel/mAdam/conv2d_32/bias/mAdam/conv2d_33/kernel/mAdam/conv2d_33/bias/mAdam/conv2d_34/kernel/mAdam/conv2d_34/bias/mAdam/conv2d_35/kernel/mAdam/conv2d_35/bias/mAdam/conv2d_36/kernel/mAdam/conv2d_36/bias/mAdam/conv2d_37/kernel/mAdam/conv2d_37/bias/m"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/conv2d_19/kernel/vAdam/conv2d_19/bias/vAdam/conv2d_20/kernel/vAdam/conv2d_20/bias/vAdam/conv2d_21/kernel/vAdam/conv2d_21/bias/vAdam/conv2d_22/kernel/vAdam/conv2d_22/bias/vAdam/conv2d_23/kernel/vAdam/conv2d_23/bias/vAdam/conv2d_24/kernel/vAdam/conv2d_24/bias/vAdam/conv2d_25/kernel/vAdam/conv2d_25/bias/vAdam/conv2d_26/kernel/vAdam/conv2d_26/bias/vAdam/conv2d_27/kernel/vAdam/conv2d_27/bias/vAdam/conv2d_28/kernel/vAdam/conv2d_28/bias/vAdam/conv2d_29/kernel/vAdam/conv2d_29/bias/vAdam/conv2d_30/kernel/vAdam/conv2d_30/bias/vAdam/conv2d_31/kernel/vAdam/conv2d_31/bias/vAdam/conv2d_32/kernel/vAdam/conv2d_32/bias/vAdam/conv2d_33/kernel/vAdam/conv2d_33/bias/vAdam/conv2d_34/kernel/vAdam/conv2d_34/bias/vAdam/conv2d_35/kernel/vAdam/conv2d_35/bias/vAdam/conv2d_36/kernel/vAdam/conv2d_36/bias/vAdam/conv2d_37/kernel/vAdam/conv2d_37/bias/v*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_39278??
?
~
)__inference_conv2d_26_layer_call_fn_38163

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_362292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_4_layer_call_fn_38216
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_4_layer_call_and_return_conditional_losses_363082
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*\
_input_shapesK
I:,????????????????????????????:?????????@:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????@
"
_user_specified_name
inputs/1
?
g
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_35915

inputs
identity?
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 2
Pad/paddings?
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Pad?
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
t
H__inference_concatenate_4_layer_call_and_return_conditional_losses_38210
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*\
_input_shapesK
I:,????????????????????????????:?????????@:l h
B
_output_shapes0
.:,????????????????????????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????@
"
_user_specified_name
inputs/1
?
g
K__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_35940

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_20_layer_call_and_return_conditional_losses_38034

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_24_layer_call_and_return_conditional_losses_36174

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?

?
D__inference_conv2d_19_layer_call_and_return_conditional_losses_38014

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_21_layer_call_fn_38063

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_360922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
g
K__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_37865

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_1_layer_call_fn_37990

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_357272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35758

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_31_layer_call_and_return_conditional_losses_38280

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  `::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  `
 
_user_specified_nameinputs
?
L
0__inference_zero_padding2d_1_layer_call_fn_35921

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_359152
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_37761

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_368632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_7_layer_call_fn_38375
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_7_layer_call_and_return_conditional_losses_365212
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+???????????????????????????:???????????:k g
A
_output_shapes/
-:+???????????????????????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?

?
D__inference_conv2d_20_layer_call_and_return_conditional_losses_36064

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_21_layer_call_and_return_conditional_losses_36092

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_35811

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_4_layer_call_fn_35781

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_357752
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_7_layer_call_fn_35817

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_358112
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
L
0__inference_gaussian_noise_1_layer_call_fn_37875

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_359402
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_6_layer_call_fn_38322
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_6_layer_call_and_return_conditional_losses_364502
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@02

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+??????????????????????????? :?????????@@:k g
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/1
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_37672

inputs1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_19_conv2d_readvariableop_resource-
)conv2d_19_biasadd_readvariableop_resource,
(conv2d_20_conv2d_readvariableop_resource-
)conv2d_20_biasadd_readvariableop_resource,
(conv2d_21_conv2d_readvariableop_resource-
)conv2d_21_biasadd_readvariableop_resource,
(conv2d_22_conv2d_readvariableop_resource-
)conv2d_22_biasadd_readvariableop_resource,
(conv2d_23_conv2d_readvariableop_resource-
)conv2d_23_biasadd_readvariableop_resource,
(conv2d_24_conv2d_readvariableop_resource-
)conv2d_24_biasadd_readvariableop_resource,
(conv2d_25_conv2d_readvariableop_resource-
)conv2d_25_biasadd_readvariableop_resource,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource,
(conv2d_28_conv2d_readvariableop_resource-
)conv2d_28_biasadd_readvariableop_resource,
(conv2d_29_conv2d_readvariableop_resource-
)conv2d_29_biasadd_readvariableop_resource,
(conv2d_30_conv2d_readvariableop_resource-
)conv2d_30_biasadd_readvariableop_resource,
(conv2d_31_conv2d_readvariableop_resource-
)conv2d_31_biasadd_readvariableop_resource,
(conv2d_32_conv2d_readvariableop_resource-
)conv2d_32_biasadd_readvariableop_resource,
(conv2d_33_conv2d_readvariableop_resource-
)conv2d_33_biasadd_readvariableop_resource,
(conv2d_34_conv2d_readvariableop_resource-
)conv2d_34_biasadd_readvariableop_resource,
(conv2d_35_conv2d_readvariableop_resource-
)conv2d_35_biasadd_readvariableop_resource,
(conv2d_36_conv2d_readvariableop_resource-
)conv2d_36_biasadd_readvariableop_resource,
(conv2d_37_conv2d_readvariableop_resource-
)conv2d_37_biasadd_readvariableop_resource
identity??5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1? conv2d_19/BiasAdd/ReadVariableOp?conv2d_19/Conv2D/ReadVariableOp? conv2d_20/BiasAdd/ReadVariableOp?conv2d_20/Conv2D/ReadVariableOp? conv2d_21/BiasAdd/ReadVariableOp?conv2d_21/Conv2D/ReadVariableOp? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp? conv2d_24/BiasAdd/ReadVariableOp?conv2d_24/Conv2D/ReadVariableOp? conv2d_25/BiasAdd/ReadVariableOp?conv2d_25/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp? conv2d_28/BiasAdd/ReadVariableOp?conv2d_28/Conv2D/ReadVariableOp? conv2d_29/BiasAdd/ReadVariableOp?conv2d_29/Conv2D/ReadVariableOp? conv2d_30/BiasAdd/ReadVariableOp?conv2d_30/Conv2D/ReadVariableOp? conv2d_31/BiasAdd/ReadVariableOp?conv2d_31/Conv2D/ReadVariableOp? conv2d_32/BiasAdd/ReadVariableOp?conv2d_32/Conv2D/ReadVariableOp? conv2d_33/BiasAdd/ReadVariableOp?conv2d_33/Conv2D/ReadVariableOp? conv2d_34/BiasAdd/ReadVariableOp?conv2d_34/Conv2D/ReadVariableOp? conv2d_35/BiasAdd/ReadVariableOp?conv2d_35/Conv2D/ReadVariableOp? conv2d_36/BiasAdd/ReadVariableOp?conv2d_36/Conv2D/ReadVariableOp? conv2d_37/BiasAdd/ReadVariableOp?conv2d_37/Conv2D/ReadVariableOp?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3inputs,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3?
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_19/Conv2D/ReadVariableOp?
conv2d_19/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_19/Conv2D?
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp?
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_19/BiasAdd?
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_19/Relu?
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_20/Conv2D/ReadVariableOp?
conv2d_20/Conv2DConv2Dconv2d_19/Relu:activations:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_20/Conv2D?
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp?
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_20/BiasAdd?
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_20/Relu?
max_pooling2d_4/MaxPoolMaxPoolconv2d_20/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool?
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_21/Conv2D/ReadVariableOp?
conv2d_21/Conv2DConv2D max_pooling2d_4/MaxPool:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_21/Conv2D?
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp?
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_21/BiasAdd~
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
conv2d_21/Relu?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_22/Conv2D/ReadVariableOp?
conv2d_22/Conv2DConv2Dconv2d_21/Relu:activations:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_22/Conv2D?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_22/BiasAdd~
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
conv2d_22/Relu?
max_pooling2d_5/MaxPoolMaxPoolconv2d_22/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPool?
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_23/Conv2D/ReadVariableOp?
conv2d_23/Conv2DConv2D max_pooling2d_5/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_23/Conv2D?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_23/BiasAdd~
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
conv2d_23/Relu?
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_24/Conv2D/ReadVariableOp?
conv2d_24/Conv2DConv2Dconv2d_23/Relu:activations:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_24/Conv2D?
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp?
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_24/BiasAdd~
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
conv2d_24/Relu?
max_pooling2d_6/MaxPoolMaxPoolconv2d_24/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool?
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_25/Conv2D/ReadVariableOp?
conv2d_25/Conv2DConv2D max_pooling2d_6/MaxPool:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_25/Conv2D?
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp?
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_25/BiasAdd~
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_25/Relu?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2Dconv2d_25/Relu:activations:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_26/BiasAdd~
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_26/Relu?
max_pooling2d_7/MaxPoolMaxPoolconv2d_26/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPool?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2D max_pooling2d_7/MaxPool:output:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_27/BiasAdd
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_27/Relu?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2DConv2Dconv2d_27/Relu:activations:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_28/Conv2D?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_28/BiasAdd
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_28/Reluz
up_sampling2d_4/ShapeShapeconv2d_28/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_4/Shape?
#up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_4/strided_slice/stack?
%up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_1?
%up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_2?
up_sampling2d_4/strided_sliceStridedSliceup_sampling2d_4/Shape:output:0,up_sampling2d_4/strided_slice/stack:output:0.up_sampling2d_4/strided_slice/stack_1:output:0.up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_4/strided_slice
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const?
up_sampling2d_4/mulMul&up_sampling2d_4/strided_slice:output:0up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul?
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_28/Relu:activations:0up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2.
,up_sampling2d_4/resize/ResizeNearestNeighborx
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_4/concat/axis?
concatenate_4/concatConcatV2=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0conv2d_26/Relu:activations:0"concatenate_4/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concatenate_4/concat?
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02!
conv2d_29/Conv2D/ReadVariableOp?
conv2d_29/Conv2DConv2Dconcatenate_4/concat:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_29/Conv2D?
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp?
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_29/BiasAdd~
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_29/Relu?
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_30/Conv2D/ReadVariableOp?
conv2d_30/Conv2DConv2Dconv2d_29/Relu:activations:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_30/Conv2D?
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_30/BiasAdd/ReadVariableOp?
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_30/BiasAdd~
conv2d_30/ReluReluconv2d_30/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_30/Reluz
up_sampling2d_5/ShapeShapeconv2d_30/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_5/Shape?
#up_sampling2d_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_5/strided_slice/stack?
%up_sampling2d_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_5/strided_slice/stack_1?
%up_sampling2d_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_5/strided_slice/stack_2?
up_sampling2d_5/strided_sliceStridedSliceup_sampling2d_5/Shape:output:0,up_sampling2d_5/strided_slice/stack:output:0.up_sampling2d_5/strided_slice/stack_1:output:0.up_sampling2d_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_5/strided_slice
up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_5/Const?
up_sampling2d_5/mulMul&up_sampling2d_5/strided_slice:output:0up_sampling2d_5/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_5/mul?
,up_sampling2d_5/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_30/Relu:activations:0up_sampling2d_5/mul:z:0*
T0*/
_output_shapes
:?????????  @*
half_pixel_centers(2.
,up_sampling2d_5/resize/ResizeNearestNeighborx
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_5/concat/axis?
concatenate_5/concatConcatV2=up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0conv2d_24/Relu:activations:0"concatenate_5/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????  `2
concatenate_5/concat?
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02!
conv2d_31/Conv2D/ReadVariableOp?
conv2d_31/Conv2DConv2Dconcatenate_5/concat:output:0'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_31/Conv2D?
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_31/BiasAdd/ReadVariableOp?
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_31/BiasAdd~
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
conv2d_31/Relu?
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_32/Conv2D/ReadVariableOp?
conv2d_32/Conv2DConv2Dconv2d_31/Relu:activations:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_32/Conv2D?
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_32/BiasAdd/ReadVariableOp?
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_32/BiasAdd~
conv2d_32/ReluReluconv2d_32/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
conv2d_32/Reluz
up_sampling2d_6/ShapeShapeconv2d_32/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_6/Shape?
#up_sampling2d_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_6/strided_slice/stack?
%up_sampling2d_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_6/strided_slice/stack_1?
%up_sampling2d_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_6/strided_slice/stack_2?
up_sampling2d_6/strided_sliceStridedSliceup_sampling2d_6/Shape:output:0,up_sampling2d_6/strided_slice/stack:output:0.up_sampling2d_6/strided_slice/stack_1:output:0.up_sampling2d_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_6/strided_slice
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_6/Const?
up_sampling2d_6/mulMul&up_sampling2d_6/strided_slice:output:0up_sampling2d_6/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_6/mul?
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_32/Relu:activations:0up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:?????????@@ *
half_pixel_centers(2.
,up_sampling2d_6/resize/ResizeNearestNeighborx
concatenate_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_6/concat/axis?
concatenate_6/concatConcatV2=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0conv2d_22/Relu:activations:0"concatenate_6/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????@@02
concatenate_6/concat?
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02!
conv2d_33/Conv2D/ReadVariableOp?
conv2d_33/Conv2DConv2Dconcatenate_6/concat:output:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_33/Conv2D?
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_33/BiasAdd/ReadVariableOp?
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_33/BiasAdd~
conv2d_33/ReluReluconv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
conv2d_33/Relu?
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_34/Conv2D/ReadVariableOp?
conv2d_34/Conv2DConv2Dconv2d_33/Relu:activations:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_34/Conv2D?
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_34/BiasAdd/ReadVariableOp?
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_34/BiasAdd~
conv2d_34/ReluReluconv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
conv2d_34/Reluz
up_sampling2d_7/ShapeShapeconv2d_34/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_7/Shape?
#up_sampling2d_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_7/strided_slice/stack?
%up_sampling2d_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_7/strided_slice/stack_1?
%up_sampling2d_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_7/strided_slice/stack_2?
up_sampling2d_7/strided_sliceStridedSliceup_sampling2d_7/Shape:output:0,up_sampling2d_7/strided_slice/stack:output:0.up_sampling2d_7/strided_slice/stack_1:output:0.up_sampling2d_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_7/strided_slice
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_7/Const?
up_sampling2d_7/mulMul&up_sampling2d_7/strided_slice:output:0up_sampling2d_7/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_7/mul?
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_34/Relu:activations:0up_sampling2d_7/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(2.
,up_sampling2d_7/resize/ResizeNearestNeighborx
concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_7/concat/axis?
concatenate_7/concatConcatV2=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0conv2d_20/Relu:activations:0"concatenate_7/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_7/concat?
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_35/Conv2D/ReadVariableOp?
conv2d_35/Conv2DConv2Dconcatenate_7/concat:output:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_35/Conv2D?
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_35/BiasAdd/ReadVariableOp?
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_35/BiasAdd?
conv2d_35/ReluReluconv2d_35/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_35/Relu?
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_36/Conv2D/ReadVariableOp?
conv2d_36/Conv2DConv2Dconv2d_35/Relu:activations:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_36/Conv2D?
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp?
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_36/BiasAdd?
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_36/Relu?
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_37/Conv2D/ReadVariableOp?
conv2d_37/Conv2DConv2Dconv2d_36/Relu:activations:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_37/Conv2D?
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp?
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_37/BiasAdd?
conv2d_37/SigmoidSigmoidconv2d_37/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_37/Sigmoid?
 cropping2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2"
 cropping2d_1/strided_slice/stack?
"cropping2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2$
"cropping2d_1/strided_slice/stack_1?
"cropping2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2$
"cropping2d_1/strided_slice/stack_2?
cropping2d_1/strided_sliceStridedSliceconv2d_37/Sigmoid:y:0)cropping2d_1/strided_slice/stack:output:0+cropping2d_1/strided_slice/stack_1:output:0+cropping2d_1/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask	*
end_mask2
cropping2d_1/strided_slice?
zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 2
zero_padding2d_1/Pad/paddings?
zero_padding2d_1/PadPad#cropping2d_1/strided_slice:output:0&zero_padding2d_1/Pad/paddings:output:0*
T0*1
_output_shapes
:???????????2
zero_padding2d_1/Pad?
IdentityIdentityzero_padding2d_1/Pad:output:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::2n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
i
0__inference_gaussian_noise_1_layer_call_fn_37870

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_359362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
f
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_35887

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_25_layer_call_and_return_conditional_losses_38134

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_22_layer_call_and_return_conditional_losses_38074

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_32_layer_call_and_return_conditional_losses_38300

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
~
)__inference_conv2d_31_layer_call_fn_38289

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_31_layer_call_and_return_conditional_losses_363992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  `::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  `
 
_user_specified_nameinputs
?

?
D__inference_conv2d_35_layer_call_and_return_conditional_losses_38386

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_19_layer_call_fn_38023

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_360372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_5_layer_call_fn_38269
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_5_layer_call_and_return_conditional_losses_363792
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  `2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+???????????????????????????@:?????????   :k g
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????   
"
_user_specified_name
inputs/1
?

?
D__inference_conv2d_35_layer_call_and_return_conditional_losses_36541

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_37977

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
r
H__inference_concatenate_5_layer_call_and_return_conditional_losses_36379

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????  `2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:?????????  `2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+???????????????????????????@:?????????   :i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
??
?8
__inference__traced_save_38863
file_prefix:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop/
+savev2_conv2d_21_kernel_read_readvariableop-
)savev2_conv2d_21_bias_read_readvariableop/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop/
+savev2_conv2d_23_kernel_read_readvariableop-
)savev2_conv2d_23_bias_read_readvariableop/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop/
+savev2_conv2d_25_kernel_read_readvariableop-
)savev2_conv2d_25_bias_read_readvariableop/
+savev2_conv2d_26_kernel_read_readvariableop-
)savev2_conv2d_26_bias_read_readvariableop/
+savev2_conv2d_27_kernel_read_readvariableop-
)savev2_conv2d_27_bias_read_readvariableop/
+savev2_conv2d_28_kernel_read_readvariableop-
)savev2_conv2d_28_bias_read_readvariableop/
+savev2_conv2d_29_kernel_read_readvariableop-
)savev2_conv2d_29_bias_read_readvariableop/
+savev2_conv2d_30_kernel_read_readvariableop-
)savev2_conv2d_30_bias_read_readvariableop/
+savev2_conv2d_31_kernel_read_readvariableop-
)savev2_conv2d_31_bias_read_readvariableop/
+savev2_conv2d_32_kernel_read_readvariableop-
)savev2_conv2d_32_bias_read_readvariableop/
+savev2_conv2d_33_kernel_read_readvariableop-
)savev2_conv2d_33_bias_read_readvariableop/
+savev2_conv2d_34_kernel_read_readvariableop-
)savev2_conv2d_34_bias_read_readvariableop/
+savev2_conv2d_35_kernel_read_readvariableop-
)savev2_conv2d_35_bias_read_readvariableop/
+savev2_conv2d_36_kernel_read_readvariableop-
)savev2_conv2d_36_bias_read_readvariableop/
+savev2_conv2d_37_kernel_read_readvariableop-
)savev2_conv2d_37_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop6
2savev2_adam_conv2d_19_kernel_m_read_readvariableop4
0savev2_adam_conv2d_19_bias_m_read_readvariableop6
2savev2_adam_conv2d_20_kernel_m_read_readvariableop4
0savev2_adam_conv2d_20_bias_m_read_readvariableop6
2savev2_adam_conv2d_21_kernel_m_read_readvariableop4
0savev2_adam_conv2d_21_bias_m_read_readvariableop6
2savev2_adam_conv2d_22_kernel_m_read_readvariableop4
0savev2_adam_conv2d_22_bias_m_read_readvariableop6
2savev2_adam_conv2d_23_kernel_m_read_readvariableop4
0savev2_adam_conv2d_23_bias_m_read_readvariableop6
2savev2_adam_conv2d_24_kernel_m_read_readvariableop4
0savev2_adam_conv2d_24_bias_m_read_readvariableop6
2savev2_adam_conv2d_25_kernel_m_read_readvariableop4
0savev2_adam_conv2d_25_bias_m_read_readvariableop6
2savev2_adam_conv2d_26_kernel_m_read_readvariableop4
0savev2_adam_conv2d_26_bias_m_read_readvariableop6
2savev2_adam_conv2d_27_kernel_m_read_readvariableop4
0savev2_adam_conv2d_27_bias_m_read_readvariableop6
2savev2_adam_conv2d_28_kernel_m_read_readvariableop4
0savev2_adam_conv2d_28_bias_m_read_readvariableop6
2savev2_adam_conv2d_29_kernel_m_read_readvariableop4
0savev2_adam_conv2d_29_bias_m_read_readvariableop6
2savev2_adam_conv2d_30_kernel_m_read_readvariableop4
0savev2_adam_conv2d_30_bias_m_read_readvariableop6
2savev2_adam_conv2d_31_kernel_m_read_readvariableop4
0savev2_adam_conv2d_31_bias_m_read_readvariableop6
2savev2_adam_conv2d_32_kernel_m_read_readvariableop4
0savev2_adam_conv2d_32_bias_m_read_readvariableop6
2savev2_adam_conv2d_33_kernel_m_read_readvariableop4
0savev2_adam_conv2d_33_bias_m_read_readvariableop6
2savev2_adam_conv2d_34_kernel_m_read_readvariableop4
0savev2_adam_conv2d_34_bias_m_read_readvariableop6
2savev2_adam_conv2d_35_kernel_m_read_readvariableop4
0savev2_adam_conv2d_35_bias_m_read_readvariableop6
2savev2_adam_conv2d_36_kernel_m_read_readvariableop4
0savev2_adam_conv2d_36_bias_m_read_readvariableop6
2savev2_adam_conv2d_37_kernel_m_read_readvariableop4
0savev2_adam_conv2d_37_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop6
2savev2_adam_conv2d_19_kernel_v_read_readvariableop4
0savev2_adam_conv2d_19_bias_v_read_readvariableop6
2savev2_adam_conv2d_20_kernel_v_read_readvariableop4
0savev2_adam_conv2d_20_bias_v_read_readvariableop6
2savev2_adam_conv2d_21_kernel_v_read_readvariableop4
0savev2_adam_conv2d_21_bias_v_read_readvariableop6
2savev2_adam_conv2d_22_kernel_v_read_readvariableop4
0savev2_adam_conv2d_22_bias_v_read_readvariableop6
2savev2_adam_conv2d_23_kernel_v_read_readvariableop4
0savev2_adam_conv2d_23_bias_v_read_readvariableop6
2savev2_adam_conv2d_24_kernel_v_read_readvariableop4
0savev2_adam_conv2d_24_bias_v_read_readvariableop6
2savev2_adam_conv2d_25_kernel_v_read_readvariableop4
0savev2_adam_conv2d_25_bias_v_read_readvariableop6
2savev2_adam_conv2d_26_kernel_v_read_readvariableop4
0savev2_adam_conv2d_26_bias_v_read_readvariableop6
2savev2_adam_conv2d_27_kernel_v_read_readvariableop4
0savev2_adam_conv2d_27_bias_v_read_readvariableop6
2savev2_adam_conv2d_28_kernel_v_read_readvariableop4
0savev2_adam_conv2d_28_bias_v_read_readvariableop6
2savev2_adam_conv2d_29_kernel_v_read_readvariableop4
0savev2_adam_conv2d_29_bias_v_read_readvariableop6
2savev2_adam_conv2d_30_kernel_v_read_readvariableop4
0savev2_adam_conv2d_30_bias_v_read_readvariableop6
2savev2_adam_conv2d_31_kernel_v_read_readvariableop4
0savev2_adam_conv2d_31_bias_v_read_readvariableop6
2savev2_adam_conv2d_32_kernel_v_read_readvariableop4
0savev2_adam_conv2d_32_bias_v_read_readvariableop6
2savev2_adam_conv2d_33_kernel_v_read_readvariableop4
0savev2_adam_conv2d_33_bias_v_read_readvariableop6
2savev2_adam_conv2d_34_kernel_v_read_readvariableop4
0savev2_adam_conv2d_34_bias_v_read_readvariableop6
2savev2_adam_conv2d_35_kernel_v_read_readvariableop4
0savev2_adam_conv2d_35_bias_v_read_readvariableop6
2savev2_adam_conv2d_36_kernel_v_read_readvariableop4
0savev2_adam_conv2d_36_bias_v_read_readvariableop6
2savev2_adam_conv2d_37_kernel_v_read_readvariableop4
0savev2_adam_conv2d_37_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?L
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?L
value?KB?K?B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?5
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop+savev2_conv2d_21_kernel_read_readvariableop)savev2_conv2d_21_bias_read_readvariableop+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop+savev2_conv2d_23_kernel_read_readvariableop)savev2_conv2d_23_bias_read_readvariableop+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop+savev2_conv2d_25_kernel_read_readvariableop)savev2_conv2d_25_bias_read_readvariableop+savev2_conv2d_26_kernel_read_readvariableop)savev2_conv2d_26_bias_read_readvariableop+savev2_conv2d_27_kernel_read_readvariableop)savev2_conv2d_27_bias_read_readvariableop+savev2_conv2d_28_kernel_read_readvariableop)savev2_conv2d_28_bias_read_readvariableop+savev2_conv2d_29_kernel_read_readvariableop)savev2_conv2d_29_bias_read_readvariableop+savev2_conv2d_30_kernel_read_readvariableop)savev2_conv2d_30_bias_read_readvariableop+savev2_conv2d_31_kernel_read_readvariableop)savev2_conv2d_31_bias_read_readvariableop+savev2_conv2d_32_kernel_read_readvariableop)savev2_conv2d_32_bias_read_readvariableop+savev2_conv2d_33_kernel_read_readvariableop)savev2_conv2d_33_bias_read_readvariableop+savev2_conv2d_34_kernel_read_readvariableop)savev2_conv2d_34_bias_read_readvariableop+savev2_conv2d_35_kernel_read_readvariableop)savev2_conv2d_35_bias_read_readvariableop+savev2_conv2d_36_kernel_read_readvariableop)savev2_conv2d_36_bias_read_readvariableop+savev2_conv2d_37_kernel_read_readvariableop)savev2_conv2d_37_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop2savev2_adam_conv2d_19_kernel_m_read_readvariableop0savev2_adam_conv2d_19_bias_m_read_readvariableop2savev2_adam_conv2d_20_kernel_m_read_readvariableop0savev2_adam_conv2d_20_bias_m_read_readvariableop2savev2_adam_conv2d_21_kernel_m_read_readvariableop0savev2_adam_conv2d_21_bias_m_read_readvariableop2savev2_adam_conv2d_22_kernel_m_read_readvariableop0savev2_adam_conv2d_22_bias_m_read_readvariableop2savev2_adam_conv2d_23_kernel_m_read_readvariableop0savev2_adam_conv2d_23_bias_m_read_readvariableop2savev2_adam_conv2d_24_kernel_m_read_readvariableop0savev2_adam_conv2d_24_bias_m_read_readvariableop2savev2_adam_conv2d_25_kernel_m_read_readvariableop0savev2_adam_conv2d_25_bias_m_read_readvariableop2savev2_adam_conv2d_26_kernel_m_read_readvariableop0savev2_adam_conv2d_26_bias_m_read_readvariableop2savev2_adam_conv2d_27_kernel_m_read_readvariableop0savev2_adam_conv2d_27_bias_m_read_readvariableop2savev2_adam_conv2d_28_kernel_m_read_readvariableop0savev2_adam_conv2d_28_bias_m_read_readvariableop2savev2_adam_conv2d_29_kernel_m_read_readvariableop0savev2_adam_conv2d_29_bias_m_read_readvariableop2savev2_adam_conv2d_30_kernel_m_read_readvariableop0savev2_adam_conv2d_30_bias_m_read_readvariableop2savev2_adam_conv2d_31_kernel_m_read_readvariableop0savev2_adam_conv2d_31_bias_m_read_readvariableop2savev2_adam_conv2d_32_kernel_m_read_readvariableop0savev2_adam_conv2d_32_bias_m_read_readvariableop2savev2_adam_conv2d_33_kernel_m_read_readvariableop0savev2_adam_conv2d_33_bias_m_read_readvariableop2savev2_adam_conv2d_34_kernel_m_read_readvariableop0savev2_adam_conv2d_34_bias_m_read_readvariableop2savev2_adam_conv2d_35_kernel_m_read_readvariableop0savev2_adam_conv2d_35_bias_m_read_readvariableop2savev2_adam_conv2d_36_kernel_m_read_readvariableop0savev2_adam_conv2d_36_bias_m_read_readvariableop2savev2_adam_conv2d_37_kernel_m_read_readvariableop0savev2_adam_conv2d_37_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop2savev2_adam_conv2d_19_kernel_v_read_readvariableop0savev2_adam_conv2d_19_bias_v_read_readvariableop2savev2_adam_conv2d_20_kernel_v_read_readvariableop0savev2_adam_conv2d_20_bias_v_read_readvariableop2savev2_adam_conv2d_21_kernel_v_read_readvariableop0savev2_adam_conv2d_21_bias_v_read_readvariableop2savev2_adam_conv2d_22_kernel_v_read_readvariableop0savev2_adam_conv2d_22_bias_v_read_readvariableop2savev2_adam_conv2d_23_kernel_v_read_readvariableop0savev2_adam_conv2d_23_bias_v_read_readvariableop2savev2_adam_conv2d_24_kernel_v_read_readvariableop0savev2_adam_conv2d_24_bias_v_read_readvariableop2savev2_adam_conv2d_25_kernel_v_read_readvariableop0savev2_adam_conv2d_25_bias_v_read_readvariableop2savev2_adam_conv2d_26_kernel_v_read_readvariableop0savev2_adam_conv2d_26_bias_v_read_readvariableop2savev2_adam_conv2d_27_kernel_v_read_readvariableop0savev2_adam_conv2d_27_bias_v_read_readvariableop2savev2_adam_conv2d_28_kernel_v_read_readvariableop0savev2_adam_conv2d_28_bias_v_read_readvariableop2savev2_adam_conv2d_29_kernel_v_read_readvariableop0savev2_adam_conv2d_29_bias_v_read_readvariableop2savev2_adam_conv2d_30_kernel_v_read_readvariableop0savev2_adam_conv2d_30_bias_v_read_readvariableop2savev2_adam_conv2d_31_kernel_v_read_readvariableop0savev2_adam_conv2d_31_bias_v_read_readvariableop2savev2_adam_conv2d_32_kernel_v_read_readvariableop0savev2_adam_conv2d_32_bias_v_read_readvariableop2savev2_adam_conv2d_33_kernel_v_read_readvariableop0savev2_adam_conv2d_33_bias_v_read_readvariableop2savev2_adam_conv2d_34_kernel_v_read_readvariableop0savev2_adam_conv2d_34_bias_v_read_readvariableop2savev2_adam_conv2d_35_kernel_v_read_readvariableop0savev2_adam_conv2d_35_bias_v_read_readvariableop2savev2_adam_conv2d_36_kernel_v_read_readvariableop0savev2_adam_conv2d_36_bias_v_read_readvariableop2savev2_adam_conv2d_37_kernel_v_read_readvariableop0savev2_adam_conv2d_37_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::::::::: : :  : : @:@:@@:@:@?:?:??:?:?@:@:@@:@:` : :  : :0:::::::::: : : : : : : : : : : : : ::::::::::: : :  : : @:@:@@:@:@?:?:??:?:?@:@:@@:@:` : :  : :0:::::::::::::::::::: : :  : : @:@:@@:@:@?:?:??:?:?@:@:@@:@:` : :  : :0:::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:-)
'
_output_shapes
:?@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:` : 

_output_shapes
: :,(
&
_output_shapes
:  :  

_output_shapes
: :,!(
&
_output_shapes
:0: "

_output_shapes
::,#(
&
_output_shapes
:: $

_output_shapes
::,%(
&
_output_shapes
:: &

_output_shapes
::,'(
&
_output_shapes
:: (

_output_shapes
::,)(
&
_output_shapes
:: *

_output_shapes
::+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: : 8

_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
::,<(
&
_output_shapes
:: =

_output_shapes
::,>(
&
_output_shapes
:: ?

_output_shapes
::,@(
&
_output_shapes
:: A

_output_shapes
::,B(
&
_output_shapes
: : C

_output_shapes
: :,D(
&
_output_shapes
:  : E

_output_shapes
: :,F(
&
_output_shapes
: @: G

_output_shapes
:@:,H(
&
_output_shapes
:@@: I

_output_shapes
:@:-J)
'
_output_shapes
:@?:!K

_output_shapes	
:?:.L*
(
_output_shapes
:??:!M

_output_shapes	
:?:-N)
'
_output_shapes
:?@: O

_output_shapes
:@:,P(
&
_output_shapes
:@@: Q

_output_shapes
:@:,R(
&
_output_shapes
:` : S

_output_shapes
: :,T(
&
_output_shapes
:  : U

_output_shapes
: :,V(
&
_output_shapes
:0: W

_output_shapes
::,X(
&
_output_shapes
:: Y

_output_shapes
::,Z(
&
_output_shapes
:: [

_output_shapes
::,\(
&
_output_shapes
:: ]

_output_shapes
::,^(
&
_output_shapes
:: _

_output_shapes
:: `

_output_shapes
:: a

_output_shapes
::,b(
&
_output_shapes
:: c

_output_shapes
::,d(
&
_output_shapes
:: e

_output_shapes
::,f(
&
_output_shapes
:: g

_output_shapes
::,h(
&
_output_shapes
:: i

_output_shapes
::,j(
&
_output_shapes
: : k

_output_shapes
: :,l(
&
_output_shapes
:  : m

_output_shapes
: :,n(
&
_output_shapes
: @: o

_output_shapes
:@:,p(
&
_output_shapes
:@@: q

_output_shapes
:@:-r)
'
_output_shapes
:@?:!s

_output_shapes	
:?:.t*
(
_output_shapes
:??:!u

_output_shapes	
:?:-v)
'
_output_shapes
:?@: w

_output_shapes
:@:,x(
&
_output_shapes
:@@: y

_output_shapes
:@:,z(
&
_output_shapes
:` : {

_output_shapes
: :,|(
&
_output_shapes
:  : }

_output_shapes
: :,~(
&
_output_shapes
:0: 

_output_shapes
::-?(
&
_output_shapes
::!?

_output_shapes
::-?(
&
_output_shapes
::!?

_output_shapes
::-?(
&
_output_shapes
::!?

_output_shapes
::-?(
&
_output_shapes
::!?

_output_shapes
::?

_output_shapes
: 
?
~
)__inference_conv2d_30_layer_call_fn_38256

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_30_layer_call_and_return_conditional_losses_363552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_35799

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_35868

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_32_layer_call_fn_38309

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_364262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_1_layer_call_fn_37939

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_359902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_37_layer_call_and_return_conditional_losses_36595

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddk
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:???????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_1_layer_call_fn_38003

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_357582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
r
H__inference_concatenate_6_layer_call_and_return_conditional_losses_36450

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????@@02
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:?????????@@02

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+??????????????????????????? :?????????@@:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_26_layer_call_and_return_conditional_losses_36229

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_38227

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
H
,__inference_cropping2d_1_layer_call_fn_35908

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_cropping2d_1_layer_call_and_return_conditional_losses_359022
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
K
/__inference_up_sampling2d_4_layer_call_fn_35836

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_358302
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_35_layer_call_fn_38395

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_365412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_35787

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_24_layer_call_and_return_conditional_losses_38114

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_37261
	rgb_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	rgb_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_356652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
1
_output_shapes
:???????????
#
_user_specified_name	RGB_Input
?

?
D__inference_conv2d_19_layer_call_and_return_conditional_losses_36037

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_34_layer_call_fn_38362

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_364972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
t
H__inference_concatenate_5_layer_call_and_return_conditional_losses_38263
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????  `2
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:?????????  `2

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+???????????????????????????@:?????????   :k g
A
_output_shapes/
-:+???????????????????????????@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????   
"
_user_specified_name
inputs/1
?
~
)__inference_conv2d_22_layer_call_fn_38083

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_361192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

j
K__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_35936

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*1
_output_shapes
:???????????*
dtype0*
seed???)*
seed2??2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*1
_output_shapes
:???????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*1
_output_shapes
:???????????2
random_normalj
addAddV2inputsrandom_normal:z:0*
T0*1
_output_shapes
:???????????2
adde
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_37162
	rgb_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	rgb_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_370752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
1
_output_shapes
:???????????
#
_user_specified_name	RGB_Input
?
f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_35775

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_37850

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*L
_read_only_resource_inputs.
,*	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_370752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_28_layer_call_fn_38203

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_362842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_27_layer_call_and_return_conditional_losses_36257

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_30_layer_call_and_return_conditional_losses_38247

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
t
H__inference_concatenate_6_layer_call_and_return_conditional_losses_38316
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????@@02
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:?????????@@02

Identity"
identityIdentity:output:0*[
_input_shapesJ
H:+??????????????????????????? :?????????@@:k g
A
_output_shapes/
-:+??????????????????????????? 
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/1
?

?
D__inference_conv2d_36_layer_call_and_return_conditional_losses_38406

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
'__inference_model_1_layer_call_fn_36950
	rgb_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	rgb_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*J
_read_only_resource_inputs,
*(	
 !"#$%&'()**0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_1_layer_call_and_return_conditional_losses_368632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
1
_output_shapes
:???????????
#
_user_specified_name	RGB_Input
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_36614
	rgb_input
batch_normalization_1_36017
batch_normalization_1_36019
batch_normalization_1_36021
batch_normalization_1_36023
conv2d_19_36048
conv2d_19_36050
conv2d_20_36075
conv2d_20_36077
conv2d_21_36103
conv2d_21_36105
conv2d_22_36130
conv2d_22_36132
conv2d_23_36158
conv2d_23_36160
conv2d_24_36185
conv2d_24_36187
conv2d_25_36213
conv2d_25_36215
conv2d_26_36240
conv2d_26_36242
conv2d_27_36268
conv2d_27_36270
conv2d_28_36295
conv2d_28_36297
conv2d_29_36339
conv2d_29_36341
conv2d_30_36366
conv2d_30_36368
conv2d_31_36410
conv2d_31_36412
conv2d_32_36437
conv2d_32_36439
conv2d_33_36481
conv2d_33_36483
conv2d_34_36508
conv2d_34_36510
conv2d_35_36552
conv2d_35_36554
conv2d_36_36579
conv2d_36_36581
conv2d_37_36606
conv2d_37_36608
identity??-batch_normalization_1/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?!conv2d_30/StatefulPartitionedCall?!conv2d_31/StatefulPartitionedCall?!conv2d_32/StatefulPartitionedCall?!conv2d_33/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall?!conv2d_35/StatefulPartitionedCall?!conv2d_36/StatefulPartitionedCall?!conv2d_37/StatefulPartitionedCall?(gaussian_noise_1/StatefulPartitionedCall?
(gaussian_noise_1/StatefulPartitionedCallStatefulPartitionedCall	rgb_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_359362*
(gaussian_noise_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_1/StatefulPartitionedCall:output:0batch_normalization_1_36017batch_normalization_1_36019batch_normalization_1_36021batch_normalization_1_36023*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_359722/
-batch_normalization_1/StatefulPartitionedCall?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_19_36048conv2d_19_36050*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_360372#
!conv2d_19/StatefulPartitionedCall?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_36075conv2d_20_36077*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_360642#
!conv2d_20/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_357752!
max_pooling2d_4/PartitionedCall?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_21_36103conv2d_21_36105*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_360922#
!conv2d_21/StatefulPartitionedCall?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_36130conv2d_22_36132*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_361192#
!conv2d_22/StatefulPartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_357872!
max_pooling2d_5/PartitionedCall?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_23_36158conv2d_23_36160*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_361472#
!conv2d_23/StatefulPartitionedCall?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0conv2d_24_36185conv2d_24_36187*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_361742#
!conv2d_24/StatefulPartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_357992!
max_pooling2d_6/PartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_25_36213conv2d_25_36215*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_362022#
!conv2d_25/StatefulPartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_26_36240conv2d_26_36242*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_362292#
!conv2d_26/StatefulPartitionedCall?
max_pooling2d_7/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_358112!
max_pooling2d_7/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_27_36268conv2d_27_36270*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_362572#
!conv2d_27/StatefulPartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_36295conv2d_28_36297*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_362842#
!conv2d_28/StatefulPartitionedCall?
up_sampling2d_4/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_358302!
up_sampling2d_4/PartitionedCall?
concatenate_4/PartitionedCallPartitionedCall(up_sampling2d_4/PartitionedCall:output:0*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_4_layer_call_and_return_conditional_losses_363082
concatenate_4/PartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_29_36339conv2d_29_36341*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_363282#
!conv2d_29/StatefulPartitionedCall?
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0conv2d_30_36366conv2d_30_36368*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_30_layer_call_and_return_conditional_losses_363552#
!conv2d_30/StatefulPartitionedCall?
up_sampling2d_5/PartitionedCallPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_358492!
up_sampling2d_5/PartitionedCall?
concatenate_5/PartitionedCallPartitionedCall(up_sampling2d_5/PartitionedCall:output:0*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_5_layer_call_and_return_conditional_losses_363792
concatenate_5/PartitionedCall?
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_31_36410conv2d_31_36412*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_31_layer_call_and_return_conditional_losses_363992#
!conv2d_31/StatefulPartitionedCall?
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0conv2d_32_36437conv2d_32_36439*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_364262#
!conv2d_32/StatefulPartitionedCall?
up_sampling2d_6/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_358682!
up_sampling2d_6/PartitionedCall?
concatenate_6/PartitionedCallPartitionedCall(up_sampling2d_6/PartitionedCall:output:0*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_6_layer_call_and_return_conditional_losses_364502
concatenate_6/PartitionedCall?
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0conv2d_33_36481conv2d_33_36483*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_364702#
!conv2d_33/StatefulPartitionedCall?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0conv2d_34_36508conv2d_34_36510*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_364972#
!conv2d_34/StatefulPartitionedCall?
up_sampling2d_7/PartitionedCallPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_358872!
up_sampling2d_7/PartitionedCall?
concatenate_7/PartitionedCallPartitionedCall(up_sampling2d_7/PartitionedCall:output:0*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_7_layer_call_and_return_conditional_losses_365212
concatenate_7/PartitionedCall?
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0conv2d_35_36552conv2d_35_36554*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_365412#
!conv2d_35/StatefulPartitionedCall?
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0conv2d_36_36579conv2d_36_36581*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_365682#
!conv2d_36/StatefulPartitionedCall?
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_36606conv2d_37_36608*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_365952#
!conv2d_37/StatefulPartitionedCall?
cropping2d_1/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_cropping2d_1_layer_call_and_return_conditional_losses_359022
cropping2d_1/PartitionedCall?
 zero_padding2d_1/PartitionedCallPartitionedCall%cropping2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_359152"
 zero_padding2d_1/PartitionedCall?
IdentityIdentity)zero_padding2d_1/PartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall)^gaussian_noise_1/StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2T
(gaussian_noise_1/StatefulPartitionedCall(gaussian_noise_1/StatefulPartitionedCall:\ X
1
_output_shapes
:???????????
#
_user_specified_name	RGB_Input
?
K
/__inference_up_sampling2d_7_layer_call_fn_35893

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_358872
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_25_layer_call_fn_38143

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_362022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_27_layer_call_and_return_conditional_losses_38174

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_34_layer_call_and_return_conditional_losses_36497

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
~
)__inference_conv2d_29_layer_call_fn_38236

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_363282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_36863

inputs
batch_normalization_1_36744
batch_normalization_1_36746
batch_normalization_1_36748
batch_normalization_1_36750
conv2d_19_36753
conv2d_19_36755
conv2d_20_36758
conv2d_20_36760
conv2d_21_36764
conv2d_21_36766
conv2d_22_36769
conv2d_22_36771
conv2d_23_36775
conv2d_23_36777
conv2d_24_36780
conv2d_24_36782
conv2d_25_36786
conv2d_25_36788
conv2d_26_36791
conv2d_26_36793
conv2d_27_36797
conv2d_27_36799
conv2d_28_36802
conv2d_28_36804
conv2d_29_36809
conv2d_29_36811
conv2d_30_36814
conv2d_30_36816
conv2d_31_36821
conv2d_31_36823
conv2d_32_36826
conv2d_32_36828
conv2d_33_36833
conv2d_33_36835
conv2d_34_36838
conv2d_34_36840
conv2d_35_36845
conv2d_35_36847
conv2d_36_36850
conv2d_36_36852
conv2d_37_36855
conv2d_37_36857
identity??-batch_normalization_1/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?!conv2d_30/StatefulPartitionedCall?!conv2d_31/StatefulPartitionedCall?!conv2d_32/StatefulPartitionedCall?!conv2d_33/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall?!conv2d_35/StatefulPartitionedCall?!conv2d_36/StatefulPartitionedCall?!conv2d_37/StatefulPartitionedCall?(gaussian_noise_1/StatefulPartitionedCall?
(gaussian_noise_1/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_359362*
(gaussian_noise_1/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1gaussian_noise_1/StatefulPartitionedCall:output:0batch_normalization_1_36744batch_normalization_1_36746batch_normalization_1_36748batch_normalization_1_36750*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_359722/
-batch_normalization_1/StatefulPartitionedCall?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_19_36753conv2d_19_36755*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_360372#
!conv2d_19/StatefulPartitionedCall?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_36758conv2d_20_36760*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_360642#
!conv2d_20/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_357752!
max_pooling2d_4/PartitionedCall?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_21_36764conv2d_21_36766*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_360922#
!conv2d_21/StatefulPartitionedCall?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_36769conv2d_22_36771*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_361192#
!conv2d_22/StatefulPartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_357872!
max_pooling2d_5/PartitionedCall?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_23_36775conv2d_23_36777*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_361472#
!conv2d_23/StatefulPartitionedCall?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0conv2d_24_36780conv2d_24_36782*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_361742#
!conv2d_24/StatefulPartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_357992!
max_pooling2d_6/PartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_25_36786conv2d_25_36788*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_362022#
!conv2d_25/StatefulPartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_26_36791conv2d_26_36793*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_362292#
!conv2d_26/StatefulPartitionedCall?
max_pooling2d_7/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_358112!
max_pooling2d_7/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_27_36797conv2d_27_36799*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_362572#
!conv2d_27/StatefulPartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_36802conv2d_28_36804*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_362842#
!conv2d_28/StatefulPartitionedCall?
up_sampling2d_4/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_358302!
up_sampling2d_4/PartitionedCall?
concatenate_4/PartitionedCallPartitionedCall(up_sampling2d_4/PartitionedCall:output:0*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_4_layer_call_and_return_conditional_losses_363082
concatenate_4/PartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_29_36809conv2d_29_36811*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_363282#
!conv2d_29/StatefulPartitionedCall?
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0conv2d_30_36814conv2d_30_36816*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_30_layer_call_and_return_conditional_losses_363552#
!conv2d_30/StatefulPartitionedCall?
up_sampling2d_5/PartitionedCallPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_358492!
up_sampling2d_5/PartitionedCall?
concatenate_5/PartitionedCallPartitionedCall(up_sampling2d_5/PartitionedCall:output:0*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_5_layer_call_and_return_conditional_losses_363792
concatenate_5/PartitionedCall?
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_31_36821conv2d_31_36823*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_31_layer_call_and_return_conditional_losses_363992#
!conv2d_31/StatefulPartitionedCall?
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0conv2d_32_36826conv2d_32_36828*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_364262#
!conv2d_32/StatefulPartitionedCall?
up_sampling2d_6/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_358682!
up_sampling2d_6/PartitionedCall?
concatenate_6/PartitionedCallPartitionedCall(up_sampling2d_6/PartitionedCall:output:0*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_6_layer_call_and_return_conditional_losses_364502
concatenate_6/PartitionedCall?
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0conv2d_33_36833conv2d_33_36835*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_364702#
!conv2d_33/StatefulPartitionedCall?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0conv2d_34_36838conv2d_34_36840*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_364972#
!conv2d_34/StatefulPartitionedCall?
up_sampling2d_7/PartitionedCallPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_358872!
up_sampling2d_7/PartitionedCall?
concatenate_7/PartitionedCallPartitionedCall(up_sampling2d_7/PartitionedCall:output:0*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_7_layer_call_and_return_conditional_losses_365212
concatenate_7/PartitionedCall?
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0conv2d_35_36845conv2d_35_36847*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_365412#
!conv2d_35/StatefulPartitionedCall?
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0conv2d_36_36850conv2d_36_36852*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_365682#
!conv2d_36/StatefulPartitionedCall?
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_36855conv2d_37_36857*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_365952#
!conv2d_37/StatefulPartitionedCall?
cropping2d_1/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_cropping2d_1_layer_call_and_return_conditional_losses_359022
cropping2d_1/PartitionedCall?
 zero_padding2d_1/PartitionedCallPartitionedCall%cropping2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_359152"
 zero_padding2d_1/PartitionedCall?
IdentityIdentity)zero_padding2d_1/PartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall)^gaussian_noise_1/StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2T
(gaussian_noise_1/StatefulPartitionedCall(gaussian_noise_1/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_38194

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_20_layer_call_fn_38043

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_360642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_36_layer_call_and_return_conditional_losses_36568

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
f
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_35830

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_30_layer_call_and_return_conditional_losses_36355

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_36737
	rgb_input
batch_normalization_1_36618
batch_normalization_1_36620
batch_normalization_1_36622
batch_normalization_1_36624
conv2d_19_36627
conv2d_19_36629
conv2d_20_36632
conv2d_20_36634
conv2d_21_36638
conv2d_21_36640
conv2d_22_36643
conv2d_22_36645
conv2d_23_36649
conv2d_23_36651
conv2d_24_36654
conv2d_24_36656
conv2d_25_36660
conv2d_25_36662
conv2d_26_36665
conv2d_26_36667
conv2d_27_36671
conv2d_27_36673
conv2d_28_36676
conv2d_28_36678
conv2d_29_36683
conv2d_29_36685
conv2d_30_36688
conv2d_30_36690
conv2d_31_36695
conv2d_31_36697
conv2d_32_36700
conv2d_32_36702
conv2d_33_36707
conv2d_33_36709
conv2d_34_36712
conv2d_34_36714
conv2d_35_36719
conv2d_35_36721
conv2d_36_36724
conv2d_36_36726
conv2d_37_36729
conv2d_37_36731
identity??-batch_normalization_1/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?!conv2d_30/StatefulPartitionedCall?!conv2d_31/StatefulPartitionedCall?!conv2d_32/StatefulPartitionedCall?!conv2d_33/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall?!conv2d_35/StatefulPartitionedCall?!conv2d_36/StatefulPartitionedCall?!conv2d_37/StatefulPartitionedCall?
 gaussian_noise_1/PartitionedCallPartitionedCall	rgb_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_359402"
 gaussian_noise_1/PartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)gaussian_noise_1/PartitionedCall:output:0batch_normalization_1_36618batch_normalization_1_36620batch_normalization_1_36622batch_normalization_1_36624*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_359902/
-batch_normalization_1/StatefulPartitionedCall?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_19_36627conv2d_19_36629*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_360372#
!conv2d_19/StatefulPartitionedCall?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_36632conv2d_20_36634*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_360642#
!conv2d_20/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_357752!
max_pooling2d_4/PartitionedCall?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_21_36638conv2d_21_36640*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_360922#
!conv2d_21/StatefulPartitionedCall?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_36643conv2d_22_36645*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_361192#
!conv2d_22/StatefulPartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_357872!
max_pooling2d_5/PartitionedCall?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_23_36649conv2d_23_36651*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_361472#
!conv2d_23/StatefulPartitionedCall?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0conv2d_24_36654conv2d_24_36656*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_361742#
!conv2d_24/StatefulPartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_357992!
max_pooling2d_6/PartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_25_36660conv2d_25_36662*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_362022#
!conv2d_25/StatefulPartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_26_36665conv2d_26_36667*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_362292#
!conv2d_26/StatefulPartitionedCall?
max_pooling2d_7/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_358112!
max_pooling2d_7/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_27_36671conv2d_27_36673*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_362572#
!conv2d_27/StatefulPartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_36676conv2d_28_36678*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_362842#
!conv2d_28/StatefulPartitionedCall?
up_sampling2d_4/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_358302!
up_sampling2d_4/PartitionedCall?
concatenate_4/PartitionedCallPartitionedCall(up_sampling2d_4/PartitionedCall:output:0*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_4_layer_call_and_return_conditional_losses_363082
concatenate_4/PartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_29_36683conv2d_29_36685*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_363282#
!conv2d_29/StatefulPartitionedCall?
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0conv2d_30_36688conv2d_30_36690*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_30_layer_call_and_return_conditional_losses_363552#
!conv2d_30/StatefulPartitionedCall?
up_sampling2d_5/PartitionedCallPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_358492!
up_sampling2d_5/PartitionedCall?
concatenate_5/PartitionedCallPartitionedCall(up_sampling2d_5/PartitionedCall:output:0*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_5_layer_call_and_return_conditional_losses_363792
concatenate_5/PartitionedCall?
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_31_36695conv2d_31_36697*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_31_layer_call_and_return_conditional_losses_363992#
!conv2d_31/StatefulPartitionedCall?
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0conv2d_32_36700conv2d_32_36702*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_364262#
!conv2d_32/StatefulPartitionedCall?
up_sampling2d_6/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_358682!
up_sampling2d_6/PartitionedCall?
concatenate_6/PartitionedCallPartitionedCall(up_sampling2d_6/PartitionedCall:output:0*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_6_layer_call_and_return_conditional_losses_364502
concatenate_6/PartitionedCall?
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0conv2d_33_36707conv2d_33_36709*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_364702#
!conv2d_33/StatefulPartitionedCall?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0conv2d_34_36712conv2d_34_36714*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_364972#
!conv2d_34/StatefulPartitionedCall?
up_sampling2d_7/PartitionedCallPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_358872!
up_sampling2d_7/PartitionedCall?
concatenate_7/PartitionedCallPartitionedCall(up_sampling2d_7/PartitionedCall:output:0*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_7_layer_call_and_return_conditional_losses_365212
concatenate_7/PartitionedCall?
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0conv2d_35_36719conv2d_35_36721*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_365412#
!conv2d_35/StatefulPartitionedCall?
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0conv2d_36_36724conv2d_36_36726*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_365682#
!conv2d_36/StatefulPartitionedCall?
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_36729conv2d_37_36731*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_365952#
!conv2d_37/StatefulPartitionedCall?
cropping2d_1/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_cropping2d_1_layer_call_and_return_conditional_losses_359022
cropping2d_1/PartitionedCall?
 zero_padding2d_1/PartitionedCallPartitionedCall%cropping2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_359152"
 zero_padding2d_1/PartitionedCall?
IdentityIdentity)zero_padding2d_1/PartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall:\ X
1
_output_shapes
:???????????
#
_user_specified_name	RGB_Input
?
K
/__inference_up_sampling2d_6_layer_call_fn_35874

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_358682
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_36328

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_37913

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_32_layer_call_and_return_conditional_losses_36426

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?

?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_36284

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_23_layer_call_and_return_conditional_losses_38094

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_37959

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_36_layer_call_fn_38415

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_365682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_37_layer_call_and_return_conditional_losses_38426

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddk
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:???????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_34_layer_call_and_return_conditional_losses_38353

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_21_layer_call_and_return_conditional_losses_38054

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
f
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_35849

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_6_layer_call_fn_35805

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_357992
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_22_layer_call_and_return_conditional_losses_36119

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_33_layer_call_and_return_conditional_losses_38333

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@0::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@0
 
_user_specified_nameinputs
?
?
5__inference_batch_normalization_1_layer_call_fn_37926

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_359722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_5_layer_call_fn_35793

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_357872
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_25_layer_call_and_return_conditional_losses_36202

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35727

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_37_layer_call_fn_38435

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_365952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_33_layer_call_and_return_conditional_losses_36470

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@0::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@0
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35990

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_31_layer_call_and_return_conditional_losses_36399

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  `::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  `
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_37895

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_33_layer_call_fn_38342

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_364702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@0::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@0
 
_user_specified_nameinputs
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_37471

inputs1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_19_conv2d_readvariableop_resource-
)conv2d_19_biasadd_readvariableop_resource,
(conv2d_20_conv2d_readvariableop_resource-
)conv2d_20_biasadd_readvariableop_resource,
(conv2d_21_conv2d_readvariableop_resource-
)conv2d_21_biasadd_readvariableop_resource,
(conv2d_22_conv2d_readvariableop_resource-
)conv2d_22_biasadd_readvariableop_resource,
(conv2d_23_conv2d_readvariableop_resource-
)conv2d_23_biasadd_readvariableop_resource,
(conv2d_24_conv2d_readvariableop_resource-
)conv2d_24_biasadd_readvariableop_resource,
(conv2d_25_conv2d_readvariableop_resource-
)conv2d_25_biasadd_readvariableop_resource,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource,
(conv2d_28_conv2d_readvariableop_resource-
)conv2d_28_biasadd_readvariableop_resource,
(conv2d_29_conv2d_readvariableop_resource-
)conv2d_29_biasadd_readvariableop_resource,
(conv2d_30_conv2d_readvariableop_resource-
)conv2d_30_biasadd_readvariableop_resource,
(conv2d_31_conv2d_readvariableop_resource-
)conv2d_31_biasadd_readvariableop_resource,
(conv2d_32_conv2d_readvariableop_resource-
)conv2d_32_biasadd_readvariableop_resource,
(conv2d_33_conv2d_readvariableop_resource-
)conv2d_33_biasadd_readvariableop_resource,
(conv2d_34_conv2d_readvariableop_resource-
)conv2d_34_biasadd_readvariableop_resource,
(conv2d_35_conv2d_readvariableop_resource-
)conv2d_35_biasadd_readvariableop_resource,
(conv2d_36_conv2d_readvariableop_resource-
)conv2d_36_biasadd_readvariableop_resource,
(conv2d_37_conv2d_readvariableop_resource-
)conv2d_37_biasadd_readvariableop_resource
identity??$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1? conv2d_19/BiasAdd/ReadVariableOp?conv2d_19/Conv2D/ReadVariableOp? conv2d_20/BiasAdd/ReadVariableOp?conv2d_20/Conv2D/ReadVariableOp? conv2d_21/BiasAdd/ReadVariableOp?conv2d_21/Conv2D/ReadVariableOp? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp? conv2d_24/BiasAdd/ReadVariableOp?conv2d_24/Conv2D/ReadVariableOp? conv2d_25/BiasAdd/ReadVariableOp?conv2d_25/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp? conv2d_28/BiasAdd/ReadVariableOp?conv2d_28/Conv2D/ReadVariableOp? conv2d_29/BiasAdd/ReadVariableOp?conv2d_29/Conv2D/ReadVariableOp? conv2d_30/BiasAdd/ReadVariableOp?conv2d_30/Conv2D/ReadVariableOp? conv2d_31/BiasAdd/ReadVariableOp?conv2d_31/Conv2D/ReadVariableOp? conv2d_32/BiasAdd/ReadVariableOp?conv2d_32/Conv2D/ReadVariableOp? conv2d_33/BiasAdd/ReadVariableOp?conv2d_33/Conv2D/ReadVariableOp? conv2d_34/BiasAdd/ReadVariableOp?conv2d_34/Conv2D/ReadVariableOp? conv2d_35/BiasAdd/ReadVariableOp?conv2d_35/Conv2D/ReadVariableOp? conv2d_36/BiasAdd/ReadVariableOp?conv2d_36/Conv2D/ReadVariableOp? conv2d_37/BiasAdd/ReadVariableOp?conv2d_37/Conv2D/ReadVariableOpf
gaussian_noise_1/ShapeShapeinputs*
T0*
_output_shapes
:2
gaussian_noise_1/Shape?
#gaussian_noise_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#gaussian_noise_1/random_normal/mean?
%gaussian_noise_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2'
%gaussian_noise_1/random_normal/stddev?
3gaussian_noise_1/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise_1/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype0*
seed???)*
seed2???25
3gaussian_noise_1/random_normal/RandomStandardNormal?
"gaussian_noise_1/random_normal/mulMul<gaussian_noise_1/random_normal/RandomStandardNormal:output:0.gaussian_noise_1/random_normal/stddev:output:0*
T0*1
_output_shapes
:???????????2$
"gaussian_noise_1/random_normal/mul?
gaussian_noise_1/random_normalAdd&gaussian_noise_1/random_normal/mul:z:0,gaussian_noise_1/random_normal/mean:output:0*
T0*1
_output_shapes
:???????????2 
gaussian_noise_1/random_normal?
gaussian_noise_1/addAddV2inputs"gaussian_noise_1/random_normal:z:0*
T0*1
_output_shapes
:???????????2
gaussian_noise_1/add?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3gaussian_noise_1/add:z:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1?
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_19/Conv2D/ReadVariableOp?
conv2d_19/Conv2DConv2D*batch_normalization_1/FusedBatchNormV3:y:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_19/Conv2D?
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp?
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_19/BiasAdd?
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_19/Relu?
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_20/Conv2D/ReadVariableOp?
conv2d_20/Conv2DConv2Dconv2d_19/Relu:activations:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_20/Conv2D?
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp?
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_20/BiasAdd?
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_20/Relu?
max_pooling2d_4/MaxPoolMaxPoolconv2d_20/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool?
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_21/Conv2D/ReadVariableOp?
conv2d_21/Conv2DConv2D max_pooling2d_4/MaxPool:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_21/Conv2D?
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp?
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_21/BiasAdd~
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
conv2d_21/Relu?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_22/Conv2D/ReadVariableOp?
conv2d_22/Conv2DConv2Dconv2d_21/Relu:activations:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_22/Conv2D?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_22/BiasAdd~
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
conv2d_22/Relu?
max_pooling2d_5/MaxPoolMaxPoolconv2d_22/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPool?
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_23/Conv2D/ReadVariableOp?
conv2d_23/Conv2DConv2D max_pooling2d_5/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_23/Conv2D?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_23/BiasAdd~
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
conv2d_23/Relu?
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_24/Conv2D/ReadVariableOp?
conv2d_24/Conv2DConv2Dconv2d_23/Relu:activations:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_24/Conv2D?
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp?
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_24/BiasAdd~
conv2d_24/ReluReluconv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
conv2d_24/Relu?
max_pooling2d_6/MaxPoolMaxPoolconv2d_24/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_6/MaxPool?
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_25/Conv2D/ReadVariableOp?
conv2d_25/Conv2DConv2D max_pooling2d_6/MaxPool:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_25/Conv2D?
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp?
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_25/BiasAdd~
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_25/Relu?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2Dconv2d_25/Relu:activations:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_26/BiasAdd~
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_26/Relu?
max_pooling2d_7/MaxPoolMaxPoolconv2d_26/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_7/MaxPool?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2D max_pooling2d_7/MaxPool:output:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_27/BiasAdd
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_27/Relu?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2DConv2Dconv2d_27/Relu:activations:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_28/Conv2D?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_28/BiasAdd
conv2d_28/ReluReluconv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_28/Reluz
up_sampling2d_4/ShapeShapeconv2d_28/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_4/Shape?
#up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_4/strided_slice/stack?
%up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_1?
%up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_4/strided_slice/stack_2?
up_sampling2d_4/strided_sliceStridedSliceup_sampling2d_4/Shape:output:0,up_sampling2d_4/strided_slice/stack:output:0.up_sampling2d_4/strided_slice/stack_1:output:0.up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_4/strided_slice
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const?
up_sampling2d_4/mulMul&up_sampling2d_4/strided_slice:output:0up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul?
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_28/Relu:activations:0up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2.
,up_sampling2d_4/resize/ResizeNearestNeighborx
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_4/concat/axis?
concatenate_4/concatConcatV2=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0conv2d_26/Relu:activations:0"concatenate_4/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concatenate_4/concat?
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02!
conv2d_29/Conv2D/ReadVariableOp?
conv2d_29/Conv2DConv2Dconcatenate_4/concat:output:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_29/Conv2D?
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp?
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_29/BiasAdd~
conv2d_29/ReluReluconv2d_29/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_29/Relu?
conv2d_30/Conv2D/ReadVariableOpReadVariableOp(conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_30/Conv2D/ReadVariableOp?
conv2d_30/Conv2DConv2Dconv2d_29/Relu:activations:0'conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_30/Conv2D?
 conv2d_30/BiasAdd/ReadVariableOpReadVariableOp)conv2d_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_30/BiasAdd/ReadVariableOp?
conv2d_30/BiasAddBiasAddconv2d_30/Conv2D:output:0(conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_30/BiasAdd~
conv2d_30/ReluReluconv2d_30/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_30/Reluz
up_sampling2d_5/ShapeShapeconv2d_30/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_5/Shape?
#up_sampling2d_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_5/strided_slice/stack?
%up_sampling2d_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_5/strided_slice/stack_1?
%up_sampling2d_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_5/strided_slice/stack_2?
up_sampling2d_5/strided_sliceStridedSliceup_sampling2d_5/Shape:output:0,up_sampling2d_5/strided_slice/stack:output:0.up_sampling2d_5/strided_slice/stack_1:output:0.up_sampling2d_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_5/strided_slice
up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_5/Const?
up_sampling2d_5/mulMul&up_sampling2d_5/strided_slice:output:0up_sampling2d_5/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_5/mul?
,up_sampling2d_5/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_30/Relu:activations:0up_sampling2d_5/mul:z:0*
T0*/
_output_shapes
:?????????  @*
half_pixel_centers(2.
,up_sampling2d_5/resize/ResizeNearestNeighborx
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_5/concat/axis?
concatenate_5/concatConcatV2=up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0conv2d_24/Relu:activations:0"concatenate_5/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????  `2
concatenate_5/concat?
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02!
conv2d_31/Conv2D/ReadVariableOp?
conv2d_31/Conv2DConv2Dconcatenate_5/concat:output:0'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_31/Conv2D?
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_31/BiasAdd/ReadVariableOp?
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_31/BiasAdd~
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
conv2d_31/Relu?
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_32/Conv2D/ReadVariableOp?
conv2d_32/Conv2DConv2Dconv2d_31/Relu:activations:0'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv2d_32/Conv2D?
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_32/BiasAdd/ReadVariableOp?
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv2d_32/BiasAdd~
conv2d_32/ReluReluconv2d_32/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
conv2d_32/Reluz
up_sampling2d_6/ShapeShapeconv2d_32/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_6/Shape?
#up_sampling2d_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_6/strided_slice/stack?
%up_sampling2d_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_6/strided_slice/stack_1?
%up_sampling2d_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_6/strided_slice/stack_2?
up_sampling2d_6/strided_sliceStridedSliceup_sampling2d_6/Shape:output:0,up_sampling2d_6/strided_slice/stack:output:0.up_sampling2d_6/strided_slice/stack_1:output:0.up_sampling2d_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_6/strided_slice
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_6/Const?
up_sampling2d_6/mulMul&up_sampling2d_6/strided_slice:output:0up_sampling2d_6/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_6/mul?
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_32/Relu:activations:0up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:?????????@@ *
half_pixel_centers(2.
,up_sampling2d_6/resize/ResizeNearestNeighborx
concatenate_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_6/concat/axis?
concatenate_6/concatConcatV2=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0conv2d_22/Relu:activations:0"concatenate_6/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????@@02
concatenate_6/concat?
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02!
conv2d_33/Conv2D/ReadVariableOp?
conv2d_33/Conv2DConv2Dconcatenate_6/concat:output:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_33/Conv2D?
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_33/BiasAdd/ReadVariableOp?
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_33/BiasAdd~
conv2d_33/ReluReluconv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
conv2d_33/Relu?
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_34/Conv2D/ReadVariableOp?
conv2d_34/Conv2DConv2Dconv2d_33/Relu:activations:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
conv2d_34/Conv2D?
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_34/BiasAdd/ReadVariableOp?
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
conv2d_34/BiasAdd~
conv2d_34/ReluReluconv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
conv2d_34/Reluz
up_sampling2d_7/ShapeShapeconv2d_34/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_7/Shape?
#up_sampling2d_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_7/strided_slice/stack?
%up_sampling2d_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_7/strided_slice/stack_1?
%up_sampling2d_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_7/strided_slice/stack_2?
up_sampling2d_7/strided_sliceStridedSliceup_sampling2d_7/Shape:output:0,up_sampling2d_7/strided_slice/stack:output:0.up_sampling2d_7/strided_slice/stack_1:output:0.up_sampling2d_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_7/strided_slice
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_7/Const?
up_sampling2d_7/mulMul&up_sampling2d_7/strided_slice:output:0up_sampling2d_7/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_7/mul?
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_34/Relu:activations:0up_sampling2d_7/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(2.
,up_sampling2d_7/resize/ResizeNearestNeighborx
concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_7/concat/axis?
concatenate_7/concatConcatV2=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0conv2d_20/Relu:activations:0"concatenate_7/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_7/concat?
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_35/Conv2D/ReadVariableOp?
conv2d_35/Conv2DConv2Dconcatenate_7/concat:output:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_35/Conv2D?
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_35/BiasAdd/ReadVariableOp?
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_35/BiasAdd?
conv2d_35/ReluReluconv2d_35/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_35/Relu?
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_36/Conv2D/ReadVariableOp?
conv2d_36/Conv2DConv2Dconv2d_35/Relu:activations:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_36/Conv2D?
 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_36/BiasAdd/ReadVariableOp?
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_36/BiasAdd?
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_36/Relu?
conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_37/Conv2D/ReadVariableOp?
conv2d_37/Conv2DConv2Dconv2d_36/Relu:activations:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d_37/Conv2D?
 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_37/BiasAdd/ReadVariableOp?
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_37/BiasAdd?
conv2d_37/SigmoidSigmoidconv2d_37/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_37/Sigmoid?
 cropping2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2"
 cropping2d_1/strided_slice/stack?
"cropping2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2$
"cropping2d_1/strided_slice/stack_1?
"cropping2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2$
"cropping2d_1/strided_slice/stack_2?
cropping2d_1/strided_sliceStridedSliceconv2d_37/Sigmoid:y:0)cropping2d_1/strided_slice/stack:output:0+cropping2d_1/strided_slice/stack_1:output:0+cropping2d_1/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask	*
end_mask2
cropping2d_1/strided_slice?
zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 2
zero_padding2d_1/Pad/paddings?
zero_padding2d_1/PadPad#cropping2d_1/strided_slice:output:0&zero_padding2d_1/Pad/paddings:output:0*
T0*1
_output_shapes
:???????????2
zero_padding2d_1/Pad?
IdentityIdentityzero_padding2d_1/Pad:output:0%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp!^conv2d_30/BiasAdd/ReadVariableOp ^conv2d_30/Conv2D/ReadVariableOp!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2D
 conv2d_30/BiasAdd/ReadVariableOp conv2d_30/BiasAdd/ReadVariableOp2B
conv2d_30/Conv2D/ReadVariableOpconv2d_30/Conv2D/ReadVariableOp2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_23_layer_call_fn_38103

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_361472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_35972

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_23_layer_call_and_return_conditional_losses_36147

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????  ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
t
H__inference_concatenate_7_layer_call_and_return_conditional_losses_38369
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+???????????????????????????:???????????:k g
A
_output_shapes/
-:+???????????????????????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?

j
K__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_37861

inputs
identity?D
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
random_normal/stddev?
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*1
_output_shapes
:???????????*
dtype0*
seed???)*
seed2???2$
"random_normal/RandomStandardNormal?
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*1
_output_shapes
:???????????2
random_normal/mul?
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*1
_output_shapes
:???????????2
random_normalj
addAddV2inputsrandom_normal:z:0*
T0*1
_output_shapes
:???????????2
adde
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_27_layer_call_fn_38183

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_362572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
r
H__inference_concatenate_4_layer_call_and_return_conditional_losses_36308

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*\
_input_shapesK
I:,????????????????????????????:?????????@:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?G
!__inference__traced_restore_39278
file_prefix0
,assignvariableop_batch_normalization_1_gamma1
-assignvariableop_1_batch_normalization_1_beta8
4assignvariableop_2_batch_normalization_1_moving_mean<
8assignvariableop_3_batch_normalization_1_moving_variance'
#assignvariableop_4_conv2d_19_kernel%
!assignvariableop_5_conv2d_19_bias'
#assignvariableop_6_conv2d_20_kernel%
!assignvariableop_7_conv2d_20_bias'
#assignvariableop_8_conv2d_21_kernel%
!assignvariableop_9_conv2d_21_bias(
$assignvariableop_10_conv2d_22_kernel&
"assignvariableop_11_conv2d_22_bias(
$assignvariableop_12_conv2d_23_kernel&
"assignvariableop_13_conv2d_23_bias(
$assignvariableop_14_conv2d_24_kernel&
"assignvariableop_15_conv2d_24_bias(
$assignvariableop_16_conv2d_25_kernel&
"assignvariableop_17_conv2d_25_bias(
$assignvariableop_18_conv2d_26_kernel&
"assignvariableop_19_conv2d_26_bias(
$assignvariableop_20_conv2d_27_kernel&
"assignvariableop_21_conv2d_27_bias(
$assignvariableop_22_conv2d_28_kernel&
"assignvariableop_23_conv2d_28_bias(
$assignvariableop_24_conv2d_29_kernel&
"assignvariableop_25_conv2d_29_bias(
$assignvariableop_26_conv2d_30_kernel&
"assignvariableop_27_conv2d_30_bias(
$assignvariableop_28_conv2d_31_kernel&
"assignvariableop_29_conv2d_31_bias(
$assignvariableop_30_conv2d_32_kernel&
"assignvariableop_31_conv2d_32_bias(
$assignvariableop_32_conv2d_33_kernel&
"assignvariableop_33_conv2d_33_bias(
$assignvariableop_34_conv2d_34_kernel&
"assignvariableop_35_conv2d_34_bias(
$assignvariableop_36_conv2d_35_kernel&
"assignvariableop_37_conv2d_35_bias(
$assignvariableop_38_conv2d_36_kernel&
"assignvariableop_39_conv2d_36_bias(
$assignvariableop_40_conv2d_37_kernel&
"assignvariableop_41_conv2d_37_bias!
assignvariableop_42_adam_iter#
assignvariableop_43_adam_beta_1#
assignvariableop_44_adam_beta_2"
assignvariableop_45_adam_decay*
&assignvariableop_46_adam_learning_rate
assignvariableop_47_total
assignvariableop_48_count
assignvariableop_49_total_1
assignvariableop_50_count_1
assignvariableop_51_total_2
assignvariableop_52_count_2
assignvariableop_53_total_3
assignvariableop_54_count_3:
6assignvariableop_55_adam_batch_normalization_1_gamma_m9
5assignvariableop_56_adam_batch_normalization_1_beta_m/
+assignvariableop_57_adam_conv2d_19_kernel_m-
)assignvariableop_58_adam_conv2d_19_bias_m/
+assignvariableop_59_adam_conv2d_20_kernel_m-
)assignvariableop_60_adam_conv2d_20_bias_m/
+assignvariableop_61_adam_conv2d_21_kernel_m-
)assignvariableop_62_adam_conv2d_21_bias_m/
+assignvariableop_63_adam_conv2d_22_kernel_m-
)assignvariableop_64_adam_conv2d_22_bias_m/
+assignvariableop_65_adam_conv2d_23_kernel_m-
)assignvariableop_66_adam_conv2d_23_bias_m/
+assignvariableop_67_adam_conv2d_24_kernel_m-
)assignvariableop_68_adam_conv2d_24_bias_m/
+assignvariableop_69_adam_conv2d_25_kernel_m-
)assignvariableop_70_adam_conv2d_25_bias_m/
+assignvariableop_71_adam_conv2d_26_kernel_m-
)assignvariableop_72_adam_conv2d_26_bias_m/
+assignvariableop_73_adam_conv2d_27_kernel_m-
)assignvariableop_74_adam_conv2d_27_bias_m/
+assignvariableop_75_adam_conv2d_28_kernel_m-
)assignvariableop_76_adam_conv2d_28_bias_m/
+assignvariableop_77_adam_conv2d_29_kernel_m-
)assignvariableop_78_adam_conv2d_29_bias_m/
+assignvariableop_79_adam_conv2d_30_kernel_m-
)assignvariableop_80_adam_conv2d_30_bias_m/
+assignvariableop_81_adam_conv2d_31_kernel_m-
)assignvariableop_82_adam_conv2d_31_bias_m/
+assignvariableop_83_adam_conv2d_32_kernel_m-
)assignvariableop_84_adam_conv2d_32_bias_m/
+assignvariableop_85_adam_conv2d_33_kernel_m-
)assignvariableop_86_adam_conv2d_33_bias_m/
+assignvariableop_87_adam_conv2d_34_kernel_m-
)assignvariableop_88_adam_conv2d_34_bias_m/
+assignvariableop_89_adam_conv2d_35_kernel_m-
)assignvariableop_90_adam_conv2d_35_bias_m/
+assignvariableop_91_adam_conv2d_36_kernel_m-
)assignvariableop_92_adam_conv2d_36_bias_m/
+assignvariableop_93_adam_conv2d_37_kernel_m-
)assignvariableop_94_adam_conv2d_37_bias_m:
6assignvariableop_95_adam_batch_normalization_1_gamma_v9
5assignvariableop_96_adam_batch_normalization_1_beta_v/
+assignvariableop_97_adam_conv2d_19_kernel_v-
)assignvariableop_98_adam_conv2d_19_bias_v/
+assignvariableop_99_adam_conv2d_20_kernel_v.
*assignvariableop_100_adam_conv2d_20_bias_v0
,assignvariableop_101_adam_conv2d_21_kernel_v.
*assignvariableop_102_adam_conv2d_21_bias_v0
,assignvariableop_103_adam_conv2d_22_kernel_v.
*assignvariableop_104_adam_conv2d_22_bias_v0
,assignvariableop_105_adam_conv2d_23_kernel_v.
*assignvariableop_106_adam_conv2d_23_bias_v0
,assignvariableop_107_adam_conv2d_24_kernel_v.
*assignvariableop_108_adam_conv2d_24_bias_v0
,assignvariableop_109_adam_conv2d_25_kernel_v.
*assignvariableop_110_adam_conv2d_25_bias_v0
,assignvariableop_111_adam_conv2d_26_kernel_v.
*assignvariableop_112_adam_conv2d_26_bias_v0
,assignvariableop_113_adam_conv2d_27_kernel_v.
*assignvariableop_114_adam_conv2d_27_bias_v0
,assignvariableop_115_adam_conv2d_28_kernel_v.
*assignvariableop_116_adam_conv2d_28_bias_v0
,assignvariableop_117_adam_conv2d_29_kernel_v.
*assignvariableop_118_adam_conv2d_29_bias_v0
,assignvariableop_119_adam_conv2d_30_kernel_v.
*assignvariableop_120_adam_conv2d_30_bias_v0
,assignvariableop_121_adam_conv2d_31_kernel_v.
*assignvariableop_122_adam_conv2d_31_bias_v0
,assignvariableop_123_adam_conv2d_32_kernel_v.
*assignvariableop_124_adam_conv2d_32_bias_v0
,assignvariableop_125_adam_conv2d_33_kernel_v.
*assignvariableop_126_adam_conv2d_33_bias_v0
,assignvariableop_127_adam_conv2d_34_kernel_v.
*assignvariableop_128_adam_conv2d_34_bias_v0
,assignvariableop_129_adam_conv2d_35_kernel_v.
*assignvariableop_130_adam_conv2d_35_bias_v0
,assignvariableop_131_adam_conv2d_36_kernel_v.
*assignvariableop_132_adam_conv2d_36_bias_v0
,assignvariableop_133_adam_conv2d_37_kernel_v.
*assignvariableop_134_adam_conv2d_37_bias_v
identity_136??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_129?AssignVariableOp_13?AssignVariableOp_130?AssignVariableOp_131?AssignVariableOp_132?AssignVariableOp_133?AssignVariableOp_134?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?L
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?L
value?KB?K?B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp,assignvariableop_batch_normalization_1_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp-assignvariableop_1_batch_normalization_1_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp4assignvariableop_2_batch_normalization_1_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp8assignvariableop_3_batch_normalization_1_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_19_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_19_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_20_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_20_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_21_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_21_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_22_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_22_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_23_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_23_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp$assignvariableop_14_conv2d_24_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_conv2d_24_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv2d_25_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv2d_25_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_26_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_26_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv2d_27_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv2d_27_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp$assignvariableop_22_conv2d_28_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp"assignvariableop_23_conv2d_28_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_29_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_29_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp$assignvariableop_26_conv2d_30_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp"assignvariableop_27_conv2d_30_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp$assignvariableop_28_conv2d_31_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp"assignvariableop_29_conv2d_31_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp$assignvariableop_30_conv2d_32_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp"assignvariableop_31_conv2d_32_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv2d_33_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp"assignvariableop_33_conv2d_33_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp$assignvariableop_34_conv2d_34_kernelIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp"assignvariableop_35_conv2d_34_biasIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp$assignvariableop_36_conv2d_35_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp"assignvariableop_37_conv2d_35_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp$assignvariableop_38_conv2d_36_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp"assignvariableop_39_conv2d_36_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp$assignvariableop_40_conv2d_37_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp"assignvariableop_41_conv2d_37_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_adam_iterIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpassignvariableop_43_adam_beta_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpassignvariableop_44_adam_beta_2Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpassignvariableop_45_adam_decayIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp&assignvariableop_46_adam_learning_rateIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpassignvariableop_47_totalIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpassignvariableop_48_countIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpassignvariableop_49_total_1Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpassignvariableop_50_count_1Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpassignvariableop_51_total_2Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpassignvariableop_52_count_2Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpassignvariableop_53_total_3Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpassignvariableop_54_count_3Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp6assignvariableop_55_adam_batch_normalization_1_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp5assignvariableop_56_adam_batch_normalization_1_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2d_19_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2d_19_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_conv2d_20_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_conv2d_20_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_conv2d_21_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_conv2d_21_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_conv2d_22_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_conv2d_22_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2d_23_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2d_23_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_conv2d_24_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_conv2d_24_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv2d_25_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv2d_25_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_conv2d_26_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_conv2d_26_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_conv2d_27_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_conv2d_27_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_conv2d_28_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_conv2d_28_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_conv2d_29_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_conv2d_29_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp+assignvariableop_79_adam_conv2d_30_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp)assignvariableop_80_adam_conv2d_30_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_conv2d_31_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_conv2d_31_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp+assignvariableop_83_adam_conv2d_32_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp)assignvariableop_84_adam_conv2d_32_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_conv2d_33_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_conv2d_33_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp+assignvariableop_87_adam_conv2d_34_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp)assignvariableop_88_adam_conv2d_34_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adam_conv2d_35_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adam_conv2d_35_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp+assignvariableop_91_adam_conv2d_36_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp)assignvariableop_92_adam_conv2d_36_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_conv2d_37_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_conv2d_37_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp6assignvariableop_95_adam_batch_normalization_1_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp5assignvariableop_96_adam_batch_normalization_1_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_conv2d_19_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_conv2d_19_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp+assignvariableop_99_adam_conv2d_20_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp*assignvariableop_100_adam_conv2d_20_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_conv2d_21_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_conv2d_21_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103?
AssignVariableOp_103AssignVariableOp,assignvariableop_103_adam_conv2d_22_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104?
AssignVariableOp_104AssignVariableOp*assignvariableop_104_adam_conv2d_22_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105?
AssignVariableOp_105AssignVariableOp,assignvariableop_105_adam_conv2d_23_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106?
AssignVariableOp_106AssignVariableOp*assignvariableop_106_adam_conv2d_23_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107?
AssignVariableOp_107AssignVariableOp,assignvariableop_107_adam_conv2d_24_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108?
AssignVariableOp_108AssignVariableOp*assignvariableop_108_adam_conv2d_24_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109?
AssignVariableOp_109AssignVariableOp,assignvariableop_109_adam_conv2d_25_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110?
AssignVariableOp_110AssignVariableOp*assignvariableop_110_adam_conv2d_25_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111?
AssignVariableOp_111AssignVariableOp,assignvariableop_111_adam_conv2d_26_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112?
AssignVariableOp_112AssignVariableOp*assignvariableop_112_adam_conv2d_26_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113?
AssignVariableOp_113AssignVariableOp,assignvariableop_113_adam_conv2d_27_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114?
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adam_conv2d_27_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115?
AssignVariableOp_115AssignVariableOp,assignvariableop_115_adam_conv2d_28_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116?
AssignVariableOp_116AssignVariableOp*assignvariableop_116_adam_conv2d_28_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117?
AssignVariableOp_117AssignVariableOp,assignvariableop_117_adam_conv2d_29_kernel_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118?
AssignVariableOp_118AssignVariableOp*assignvariableop_118_adam_conv2d_29_bias_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119?
AssignVariableOp_119AssignVariableOp,assignvariableop_119_adam_conv2d_30_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120?
AssignVariableOp_120AssignVariableOp*assignvariableop_120_adam_conv2d_30_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121?
AssignVariableOp_121AssignVariableOp,assignvariableop_121_adam_conv2d_31_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122?
AssignVariableOp_122AssignVariableOp*assignvariableop_122_adam_conv2d_31_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123?
AssignVariableOp_123AssignVariableOp,assignvariableop_123_adam_conv2d_32_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124?
AssignVariableOp_124AssignVariableOp*assignvariableop_124_adam_conv2d_32_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125?
AssignVariableOp_125AssignVariableOp,assignvariableop_125_adam_conv2d_33_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126?
AssignVariableOp_126AssignVariableOp*assignvariableop_126_adam_conv2d_33_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127?
AssignVariableOp_127AssignVariableOp,assignvariableop_127_adam_conv2d_34_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128?
AssignVariableOp_128AssignVariableOp*assignvariableop_128_adam_conv2d_34_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129?
AssignVariableOp_129AssignVariableOp,assignvariableop_129_adam_conv2d_35_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130?
AssignVariableOp_130AssignVariableOp*assignvariableop_130_adam_conv2d_35_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131?
AssignVariableOp_131AssignVariableOp,assignvariableop_131_adam_conv2d_36_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132?
AssignVariableOp_132AssignVariableOp*assignvariableop_132_adam_conv2d_36_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133?
AssignVariableOp_133AssignVariableOp,assignvariableop_133_adam_conv2d_37_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134?
AssignVariableOp_134AssignVariableOp*assignvariableop_134_adam_conv2d_37_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1349
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_135Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_135?
Identity_136IdentityIdentity_135:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_136"%
identity_136Identity_136:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
r
H__inference_concatenate_7_layer_call_and_return_conditional_losses_36521

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*]
_input_shapesL
J:+???????????????????????????:???????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_24_layer_call_fn_38123

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_361742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????   ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????   
 
_user_specified_nameinputs
?
K
/__inference_up_sampling2d_5_layer_call_fn_35855

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_358492
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?!
 __inference__wrapped_model_35665
	rgb_input9
5model_1_batch_normalization_1_readvariableop_resource;
7model_1_batch_normalization_1_readvariableop_1_resourceJ
Fmodel_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceL
Hmodel_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource4
0model_1_conv2d_19_conv2d_readvariableop_resource5
1model_1_conv2d_19_biasadd_readvariableop_resource4
0model_1_conv2d_20_conv2d_readvariableop_resource5
1model_1_conv2d_20_biasadd_readvariableop_resource4
0model_1_conv2d_21_conv2d_readvariableop_resource5
1model_1_conv2d_21_biasadd_readvariableop_resource4
0model_1_conv2d_22_conv2d_readvariableop_resource5
1model_1_conv2d_22_biasadd_readvariableop_resource4
0model_1_conv2d_23_conv2d_readvariableop_resource5
1model_1_conv2d_23_biasadd_readvariableop_resource4
0model_1_conv2d_24_conv2d_readvariableop_resource5
1model_1_conv2d_24_biasadd_readvariableop_resource4
0model_1_conv2d_25_conv2d_readvariableop_resource5
1model_1_conv2d_25_biasadd_readvariableop_resource4
0model_1_conv2d_26_conv2d_readvariableop_resource5
1model_1_conv2d_26_biasadd_readvariableop_resource4
0model_1_conv2d_27_conv2d_readvariableop_resource5
1model_1_conv2d_27_biasadd_readvariableop_resource4
0model_1_conv2d_28_conv2d_readvariableop_resource5
1model_1_conv2d_28_biasadd_readvariableop_resource4
0model_1_conv2d_29_conv2d_readvariableop_resource5
1model_1_conv2d_29_biasadd_readvariableop_resource4
0model_1_conv2d_30_conv2d_readvariableop_resource5
1model_1_conv2d_30_biasadd_readvariableop_resource4
0model_1_conv2d_31_conv2d_readvariableop_resource5
1model_1_conv2d_31_biasadd_readvariableop_resource4
0model_1_conv2d_32_conv2d_readvariableop_resource5
1model_1_conv2d_32_biasadd_readvariableop_resource4
0model_1_conv2d_33_conv2d_readvariableop_resource5
1model_1_conv2d_33_biasadd_readvariableop_resource4
0model_1_conv2d_34_conv2d_readvariableop_resource5
1model_1_conv2d_34_biasadd_readvariableop_resource4
0model_1_conv2d_35_conv2d_readvariableop_resource5
1model_1_conv2d_35_biasadd_readvariableop_resource4
0model_1_conv2d_36_conv2d_readvariableop_resource5
1model_1_conv2d_36_biasadd_readvariableop_resource4
0model_1_conv2d_37_conv2d_readvariableop_resource5
1model_1_conv2d_37_biasadd_readvariableop_resource
identity??=model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp??model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?,model_1/batch_normalization_1/ReadVariableOp?.model_1/batch_normalization_1/ReadVariableOp_1?(model_1/conv2d_19/BiasAdd/ReadVariableOp?'model_1/conv2d_19/Conv2D/ReadVariableOp?(model_1/conv2d_20/BiasAdd/ReadVariableOp?'model_1/conv2d_20/Conv2D/ReadVariableOp?(model_1/conv2d_21/BiasAdd/ReadVariableOp?'model_1/conv2d_21/Conv2D/ReadVariableOp?(model_1/conv2d_22/BiasAdd/ReadVariableOp?'model_1/conv2d_22/Conv2D/ReadVariableOp?(model_1/conv2d_23/BiasAdd/ReadVariableOp?'model_1/conv2d_23/Conv2D/ReadVariableOp?(model_1/conv2d_24/BiasAdd/ReadVariableOp?'model_1/conv2d_24/Conv2D/ReadVariableOp?(model_1/conv2d_25/BiasAdd/ReadVariableOp?'model_1/conv2d_25/Conv2D/ReadVariableOp?(model_1/conv2d_26/BiasAdd/ReadVariableOp?'model_1/conv2d_26/Conv2D/ReadVariableOp?(model_1/conv2d_27/BiasAdd/ReadVariableOp?'model_1/conv2d_27/Conv2D/ReadVariableOp?(model_1/conv2d_28/BiasAdd/ReadVariableOp?'model_1/conv2d_28/Conv2D/ReadVariableOp?(model_1/conv2d_29/BiasAdd/ReadVariableOp?'model_1/conv2d_29/Conv2D/ReadVariableOp?(model_1/conv2d_30/BiasAdd/ReadVariableOp?'model_1/conv2d_30/Conv2D/ReadVariableOp?(model_1/conv2d_31/BiasAdd/ReadVariableOp?'model_1/conv2d_31/Conv2D/ReadVariableOp?(model_1/conv2d_32/BiasAdd/ReadVariableOp?'model_1/conv2d_32/Conv2D/ReadVariableOp?(model_1/conv2d_33/BiasAdd/ReadVariableOp?'model_1/conv2d_33/Conv2D/ReadVariableOp?(model_1/conv2d_34/BiasAdd/ReadVariableOp?'model_1/conv2d_34/Conv2D/ReadVariableOp?(model_1/conv2d_35/BiasAdd/ReadVariableOp?'model_1/conv2d_35/Conv2D/ReadVariableOp?(model_1/conv2d_36/BiasAdd/ReadVariableOp?'model_1/conv2d_36/Conv2D/ReadVariableOp?(model_1/conv2d_37/BiasAdd/ReadVariableOp?'model_1/conv2d_37/Conv2D/ReadVariableOp?
,model_1/batch_normalization_1/ReadVariableOpReadVariableOp5model_1_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype02.
,model_1/batch_normalization_1/ReadVariableOp?
.model_1/batch_normalization_1/ReadVariableOp_1ReadVariableOp7model_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype020
.model_1/batch_normalization_1/ReadVariableOp_1?
=model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpFmodel_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02?
=model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
?model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHmodel_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02A
?model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
.model_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3	rgb_input4model_1/batch_normalization_1/ReadVariableOp:value:06model_1/batch_normalization_1/ReadVariableOp_1:value:0Emodel_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Gmodel_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 20
.model_1/batch_normalization_1/FusedBatchNormV3?
'model_1/conv2d_19/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_1/conv2d_19/Conv2D/ReadVariableOp?
model_1/conv2d_19/Conv2DConv2D2model_1/batch_normalization_1/FusedBatchNormV3:y:0/model_1/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
model_1/conv2d_19/Conv2D?
(model_1/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_1/conv2d_19/BiasAdd/ReadVariableOp?
model_1/conv2d_19/BiasAddBiasAdd!model_1/conv2d_19/Conv2D:output:00model_1/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
model_1/conv2d_19/BiasAdd?
model_1/conv2d_19/ReluRelu"model_1/conv2d_19/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
model_1/conv2d_19/Relu?
'model_1/conv2d_20/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_1/conv2d_20/Conv2D/ReadVariableOp?
model_1/conv2d_20/Conv2DConv2D$model_1/conv2d_19/Relu:activations:0/model_1/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
model_1/conv2d_20/Conv2D?
(model_1/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_1/conv2d_20/BiasAdd/ReadVariableOp?
model_1/conv2d_20/BiasAddBiasAdd!model_1/conv2d_20/Conv2D:output:00model_1/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
model_1/conv2d_20/BiasAdd?
model_1/conv2d_20/ReluRelu"model_1/conv2d_20/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
model_1/conv2d_20/Relu?
model_1/max_pooling2d_4/MaxPoolMaxPool$model_1/conv2d_20/Relu:activations:0*/
_output_shapes
:?????????@@*
ksize
*
paddingVALID*
strides
2!
model_1/max_pooling2d_4/MaxPool?
'model_1/conv2d_21/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_1/conv2d_21/Conv2D/ReadVariableOp?
model_1/conv2d_21/Conv2DConv2D(model_1/max_pooling2d_4/MaxPool:output:0/model_1/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
model_1/conv2d_21/Conv2D?
(model_1/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_1/conv2d_21/BiasAdd/ReadVariableOp?
model_1/conv2d_21/BiasAddBiasAdd!model_1/conv2d_21/Conv2D:output:00model_1/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
model_1/conv2d_21/BiasAdd?
model_1/conv2d_21/ReluRelu"model_1/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
model_1/conv2d_21/Relu?
'model_1/conv2d_22/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_1/conv2d_22/Conv2D/ReadVariableOp?
model_1/conv2d_22/Conv2DConv2D$model_1/conv2d_21/Relu:activations:0/model_1/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
model_1/conv2d_22/Conv2D?
(model_1/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_1/conv2d_22/BiasAdd/ReadVariableOp?
model_1/conv2d_22/BiasAddBiasAdd!model_1/conv2d_22/Conv2D:output:00model_1/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
model_1/conv2d_22/BiasAdd?
model_1/conv2d_22/ReluRelu"model_1/conv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
model_1/conv2d_22/Relu?
model_1/max_pooling2d_5/MaxPoolMaxPool$model_1/conv2d_22/Relu:activations:0*/
_output_shapes
:?????????  *
ksize
*
paddingVALID*
strides
2!
model_1/max_pooling2d_5/MaxPool?
'model_1/conv2d_23/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_1/conv2d_23/Conv2D/ReadVariableOp?
model_1/conv2d_23/Conv2DConv2D(model_1/max_pooling2d_5/MaxPool:output:0/model_1/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
model_1/conv2d_23/Conv2D?
(model_1/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_1/conv2d_23/BiasAdd/ReadVariableOp?
model_1/conv2d_23/BiasAddBiasAdd!model_1/conv2d_23/Conv2D:output:00model_1/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
model_1/conv2d_23/BiasAdd?
model_1/conv2d_23/ReluRelu"model_1/conv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
model_1/conv2d_23/Relu?
'model_1/conv2d_24/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_1/conv2d_24/Conv2D/ReadVariableOp?
model_1/conv2d_24/Conv2DConv2D$model_1/conv2d_23/Relu:activations:0/model_1/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
model_1/conv2d_24/Conv2D?
(model_1/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_1/conv2d_24/BiasAdd/ReadVariableOp?
model_1/conv2d_24/BiasAddBiasAdd!model_1/conv2d_24/Conv2D:output:00model_1/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
model_1/conv2d_24/BiasAdd?
model_1/conv2d_24/ReluRelu"model_1/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
model_1/conv2d_24/Relu?
model_1/max_pooling2d_6/MaxPoolMaxPool$model_1/conv2d_24/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2!
model_1/max_pooling2d_6/MaxPool?
'model_1/conv2d_25/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'model_1/conv2d_25/Conv2D/ReadVariableOp?
model_1/conv2d_25/Conv2DConv2D(model_1/max_pooling2d_6/MaxPool:output:0/model_1/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model_1/conv2d_25/Conv2D?
(model_1/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_1/conv2d_25/BiasAdd/ReadVariableOp?
model_1/conv2d_25/BiasAddBiasAdd!model_1/conv2d_25/Conv2D:output:00model_1/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model_1/conv2d_25/BiasAdd?
model_1/conv2d_25/ReluRelu"model_1/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model_1/conv2d_25/Relu?
'model_1/conv2d_26/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'model_1/conv2d_26/Conv2D/ReadVariableOp?
model_1/conv2d_26/Conv2DConv2D$model_1/conv2d_25/Relu:activations:0/model_1/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model_1/conv2d_26/Conv2D?
(model_1/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_1/conv2d_26/BiasAdd/ReadVariableOp?
model_1/conv2d_26/BiasAddBiasAdd!model_1/conv2d_26/Conv2D:output:00model_1/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model_1/conv2d_26/BiasAdd?
model_1/conv2d_26/ReluRelu"model_1/conv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model_1/conv2d_26/Relu?
model_1/max_pooling2d_7/MaxPoolMaxPool$model_1/conv2d_26/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2!
model_1/max_pooling2d_7/MaxPool?
'model_1/conv2d_27/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_27_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02)
'model_1/conv2d_27/Conv2D/ReadVariableOp?
model_1/conv2d_27/Conv2DConv2D(model_1/max_pooling2d_7/MaxPool:output:0/model_1/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_1/conv2d_27/Conv2D?
(model_1/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_27_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_1/conv2d_27/BiasAdd/ReadVariableOp?
model_1/conv2d_27/BiasAddBiasAdd!model_1/conv2d_27/Conv2D:output:00model_1/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_1/conv2d_27/BiasAdd?
model_1/conv2d_27/ReluRelu"model_1/conv2d_27/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_1/conv2d_27/Relu?
'model_1/conv2d_28/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_28_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'model_1/conv2d_28/Conv2D/ReadVariableOp?
model_1/conv2d_28/Conv2DConv2D$model_1/conv2d_27/Relu:activations:0/model_1/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
model_1/conv2d_28/Conv2D?
(model_1/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_28_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(model_1/conv2d_28/BiasAdd/ReadVariableOp?
model_1/conv2d_28/BiasAddBiasAdd!model_1/conv2d_28/Conv2D:output:00model_1/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model_1/conv2d_28/BiasAdd?
model_1/conv2d_28/ReluRelu"model_1/conv2d_28/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_1/conv2d_28/Relu?
model_1/up_sampling2d_4/ShapeShape$model_1/conv2d_28/Relu:activations:0*
T0*
_output_shapes
:2
model_1/up_sampling2d_4/Shape?
+model_1/up_sampling2d_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model_1/up_sampling2d_4/strided_slice/stack?
-model_1/up_sampling2d_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/up_sampling2d_4/strided_slice/stack_1?
-model_1/up_sampling2d_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/up_sampling2d_4/strided_slice/stack_2?
%model_1/up_sampling2d_4/strided_sliceStridedSlice&model_1/up_sampling2d_4/Shape:output:04model_1/up_sampling2d_4/strided_slice/stack:output:06model_1/up_sampling2d_4/strided_slice/stack_1:output:06model_1/up_sampling2d_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%model_1/up_sampling2d_4/strided_slice?
model_1/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_1/up_sampling2d_4/Const?
model_1/up_sampling2d_4/mulMul.model_1/up_sampling2d_4/strided_slice:output:0&model_1/up_sampling2d_4/Const:output:0*
T0*
_output_shapes
:2
model_1/up_sampling2d_4/mul?
4model_1/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor$model_1/conv2d_28/Relu:activations:0model_1/up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(26
4model_1/up_sampling2d_4/resize/ResizeNearestNeighbor?
!model_1/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_4/concat/axis?
model_1/concatenate_4/concatConcatV2Emodel_1/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0$model_1/conv2d_26/Relu:activations:0*model_1/concatenate_4/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
model_1/concatenate_4/concat?
'model_1/conv2d_29/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_29_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02)
'model_1/conv2d_29/Conv2D/ReadVariableOp?
model_1/conv2d_29/Conv2DConv2D%model_1/concatenate_4/concat:output:0/model_1/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model_1/conv2d_29/Conv2D?
(model_1/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_29_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_1/conv2d_29/BiasAdd/ReadVariableOp?
model_1/conv2d_29/BiasAddBiasAdd!model_1/conv2d_29/Conv2D:output:00model_1/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model_1/conv2d_29/BiasAdd?
model_1/conv2d_29/ReluRelu"model_1/conv2d_29/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model_1/conv2d_29/Relu?
'model_1/conv2d_30/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_30_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'model_1/conv2d_30/Conv2D/ReadVariableOp?
model_1/conv2d_30/Conv2DConv2D$model_1/conv2d_29/Relu:activations:0/model_1/conv2d_30/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model_1/conv2d_30/Conv2D?
(model_1/conv2d_30/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_1/conv2d_30/BiasAdd/ReadVariableOp?
model_1/conv2d_30/BiasAddBiasAdd!model_1/conv2d_30/Conv2D:output:00model_1/conv2d_30/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model_1/conv2d_30/BiasAdd?
model_1/conv2d_30/ReluRelu"model_1/conv2d_30/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model_1/conv2d_30/Relu?
model_1/up_sampling2d_5/ShapeShape$model_1/conv2d_30/Relu:activations:0*
T0*
_output_shapes
:2
model_1/up_sampling2d_5/Shape?
+model_1/up_sampling2d_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model_1/up_sampling2d_5/strided_slice/stack?
-model_1/up_sampling2d_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/up_sampling2d_5/strided_slice/stack_1?
-model_1/up_sampling2d_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/up_sampling2d_5/strided_slice/stack_2?
%model_1/up_sampling2d_5/strided_sliceStridedSlice&model_1/up_sampling2d_5/Shape:output:04model_1/up_sampling2d_5/strided_slice/stack:output:06model_1/up_sampling2d_5/strided_slice/stack_1:output:06model_1/up_sampling2d_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%model_1/up_sampling2d_5/strided_slice?
model_1/up_sampling2d_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_1/up_sampling2d_5/Const?
model_1/up_sampling2d_5/mulMul.model_1/up_sampling2d_5/strided_slice:output:0&model_1/up_sampling2d_5/Const:output:0*
T0*
_output_shapes
:2
model_1/up_sampling2d_5/mul?
4model_1/up_sampling2d_5/resize/ResizeNearestNeighborResizeNearestNeighbor$model_1/conv2d_30/Relu:activations:0model_1/up_sampling2d_5/mul:z:0*
T0*/
_output_shapes
:?????????  @*
half_pixel_centers(26
4model_1/up_sampling2d_5/resize/ResizeNearestNeighbor?
!model_1/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_5/concat/axis?
model_1/concatenate_5/concatConcatV2Emodel_1/up_sampling2d_5/resize/ResizeNearestNeighbor:resized_images:0$model_1/conv2d_24/Relu:activations:0*model_1/concatenate_5/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????  `2
model_1/concatenate_5/concat?
'model_1/conv2d_31/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
:` *
dtype02)
'model_1/conv2d_31/Conv2D/ReadVariableOp?
model_1/conv2d_31/Conv2DConv2D%model_1/concatenate_5/concat:output:0/model_1/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
model_1/conv2d_31/Conv2D?
(model_1/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_1/conv2d_31/BiasAdd/ReadVariableOp?
model_1/conv2d_31/BiasAddBiasAdd!model_1/conv2d_31/Conv2D:output:00model_1/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
model_1/conv2d_31/BiasAdd?
model_1/conv2d_31/ReluRelu"model_1/conv2d_31/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
model_1/conv2d_31/Relu?
'model_1/conv2d_32/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'model_1/conv2d_32/Conv2D/ReadVariableOp?
model_1/conv2d_32/Conv2DConv2D$model_1/conv2d_31/Relu:activations:0/model_1/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
model_1/conv2d_32/Conv2D?
(model_1/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_1/conv2d_32/BiasAdd/ReadVariableOp?
model_1/conv2d_32/BiasAddBiasAdd!model_1/conv2d_32/Conv2D:output:00model_1/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
model_1/conv2d_32/BiasAdd?
model_1/conv2d_32/ReluRelu"model_1/conv2d_32/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
model_1/conv2d_32/Relu?
model_1/up_sampling2d_6/ShapeShape$model_1/conv2d_32/Relu:activations:0*
T0*
_output_shapes
:2
model_1/up_sampling2d_6/Shape?
+model_1/up_sampling2d_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model_1/up_sampling2d_6/strided_slice/stack?
-model_1/up_sampling2d_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/up_sampling2d_6/strided_slice/stack_1?
-model_1/up_sampling2d_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/up_sampling2d_6/strided_slice/stack_2?
%model_1/up_sampling2d_6/strided_sliceStridedSlice&model_1/up_sampling2d_6/Shape:output:04model_1/up_sampling2d_6/strided_slice/stack:output:06model_1/up_sampling2d_6/strided_slice/stack_1:output:06model_1/up_sampling2d_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%model_1/up_sampling2d_6/strided_slice?
model_1/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_1/up_sampling2d_6/Const?
model_1/up_sampling2d_6/mulMul.model_1/up_sampling2d_6/strided_slice:output:0&model_1/up_sampling2d_6/Const:output:0*
T0*
_output_shapes
:2
model_1/up_sampling2d_6/mul?
4model_1/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor$model_1/conv2d_32/Relu:activations:0model_1/up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:?????????@@ *
half_pixel_centers(26
4model_1/up_sampling2d_6/resize/ResizeNearestNeighbor?
!model_1/concatenate_6/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_6/concat/axis?
model_1/concatenate_6/concatConcatV2Emodel_1/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0$model_1/conv2d_22/Relu:activations:0*model_1/concatenate_6/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????@@02
model_1/concatenate_6/concat?
'model_1/conv2d_33/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype02)
'model_1/conv2d_33/Conv2D/ReadVariableOp?
model_1/conv2d_33/Conv2DConv2D%model_1/concatenate_6/concat:output:0/model_1/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
model_1/conv2d_33/Conv2D?
(model_1/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_1/conv2d_33/BiasAdd/ReadVariableOp?
model_1/conv2d_33/BiasAddBiasAdd!model_1/conv2d_33/Conv2D:output:00model_1/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
model_1/conv2d_33/BiasAdd?
model_1/conv2d_33/ReluRelu"model_1/conv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
model_1/conv2d_33/Relu?
'model_1/conv2d_34/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_1/conv2d_34/Conv2D/ReadVariableOp?
model_1/conv2d_34/Conv2DConv2D$model_1/conv2d_33/Relu:activations:0/model_1/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingSAME*
strides
2
model_1/conv2d_34/Conv2D?
(model_1/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_1/conv2d_34/BiasAdd/ReadVariableOp?
model_1/conv2d_34/BiasAddBiasAdd!model_1/conv2d_34/Conv2D:output:00model_1/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
model_1/conv2d_34/BiasAdd?
model_1/conv2d_34/ReluRelu"model_1/conv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
model_1/conv2d_34/Relu?
model_1/up_sampling2d_7/ShapeShape$model_1/conv2d_34/Relu:activations:0*
T0*
_output_shapes
:2
model_1/up_sampling2d_7/Shape?
+model_1/up_sampling2d_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model_1/up_sampling2d_7/strided_slice/stack?
-model_1/up_sampling2d_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/up_sampling2d_7/strided_slice/stack_1?
-model_1/up_sampling2d_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_1/up_sampling2d_7/strided_slice/stack_2?
%model_1/up_sampling2d_7/strided_sliceStridedSlice&model_1/up_sampling2d_7/Shape:output:04model_1/up_sampling2d_7/strided_slice/stack:output:06model_1/up_sampling2d_7/strided_slice/stack_1:output:06model_1/up_sampling2d_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%model_1/up_sampling2d_7/strided_slice?
model_1/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_1/up_sampling2d_7/Const?
model_1/up_sampling2d_7/mulMul.model_1/up_sampling2d_7/strided_slice:output:0&model_1/up_sampling2d_7/Const:output:0*
T0*
_output_shapes
:2
model_1/up_sampling2d_7/mul?
4model_1/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor$model_1/conv2d_34/Relu:activations:0model_1/up_sampling2d_7/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(26
4model_1/up_sampling2d_7/resize/ResizeNearestNeighbor?
!model_1/concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_7/concat/axis?
model_1/concatenate_7/concatConcatV2Emodel_1/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0$model_1/conv2d_20/Relu:activations:0*model_1/concatenate_7/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
model_1/concatenate_7/concat?
'model_1/conv2d_35/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_1/conv2d_35/Conv2D/ReadVariableOp?
model_1/conv2d_35/Conv2DConv2D%model_1/concatenate_7/concat:output:0/model_1/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
model_1/conv2d_35/Conv2D?
(model_1/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_1/conv2d_35/BiasAdd/ReadVariableOp?
model_1/conv2d_35/BiasAddBiasAdd!model_1/conv2d_35/Conv2D:output:00model_1/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
model_1/conv2d_35/BiasAdd?
model_1/conv2d_35/ReluRelu"model_1/conv2d_35/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
model_1/conv2d_35/Relu?
'model_1/conv2d_36/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_36_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_1/conv2d_36/Conv2D/ReadVariableOp?
model_1/conv2d_36/Conv2DConv2D$model_1/conv2d_35/Relu:activations:0/model_1/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
model_1/conv2d_36/Conv2D?
(model_1/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_1/conv2d_36/BiasAdd/ReadVariableOp?
model_1/conv2d_36/BiasAddBiasAdd!model_1/conv2d_36/Conv2D:output:00model_1/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
model_1/conv2d_36/BiasAdd?
model_1/conv2d_36/ReluRelu"model_1/conv2d_36/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
model_1/conv2d_36/Relu?
'model_1/conv2d_37/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_37_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'model_1/conv2d_37/Conv2D/ReadVariableOp?
model_1/conv2d_37/Conv2DConv2D$model_1/conv2d_36/Relu:activations:0/model_1/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
model_1/conv2d_37/Conv2D?
(model_1/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_1/conv2d_37/BiasAdd/ReadVariableOp?
model_1/conv2d_37/BiasAddBiasAdd!model_1/conv2d_37/Conv2D:output:00model_1/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
model_1/conv2d_37/BiasAdd?
model_1/conv2d_37/SigmoidSigmoid"model_1/conv2d_37/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
model_1/conv2d_37/Sigmoid?
(model_1/cropping2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2*
(model_1/cropping2d_1/strided_slice/stack?
*model_1/cropping2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2,
*model_1/cropping2d_1/strided_slice/stack_1?
*model_1/cropping2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2,
*model_1/cropping2d_1/strided_slice/stack_2?
"model_1/cropping2d_1/strided_sliceStridedSlicemodel_1/conv2d_37/Sigmoid:y:01model_1/cropping2d_1/strided_slice/stack:output:03model_1/cropping2d_1/strided_slice/stack_1:output:03model_1/cropping2d_1/strided_slice/stack_2:output:0*
Index0*
T0*1
_output_shapes
:???????????*

begin_mask	*
end_mask2$
"model_1/cropping2d_1/strided_slice?
%model_1/zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 2'
%model_1/zero_padding2d_1/Pad/paddings?
model_1/zero_padding2d_1/PadPad+model_1/cropping2d_1/strided_slice:output:0.model_1/zero_padding2d_1/Pad/paddings:output:0*
T0*1
_output_shapes
:???????????2
model_1/zero_padding2d_1/Pad?
IdentityIdentity%model_1/zero_padding2d_1/Pad:output:0>^model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@^model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1-^model_1/batch_normalization_1/ReadVariableOp/^model_1/batch_normalization_1/ReadVariableOp_1)^model_1/conv2d_19/BiasAdd/ReadVariableOp(^model_1/conv2d_19/Conv2D/ReadVariableOp)^model_1/conv2d_20/BiasAdd/ReadVariableOp(^model_1/conv2d_20/Conv2D/ReadVariableOp)^model_1/conv2d_21/BiasAdd/ReadVariableOp(^model_1/conv2d_21/Conv2D/ReadVariableOp)^model_1/conv2d_22/BiasAdd/ReadVariableOp(^model_1/conv2d_22/Conv2D/ReadVariableOp)^model_1/conv2d_23/BiasAdd/ReadVariableOp(^model_1/conv2d_23/Conv2D/ReadVariableOp)^model_1/conv2d_24/BiasAdd/ReadVariableOp(^model_1/conv2d_24/Conv2D/ReadVariableOp)^model_1/conv2d_25/BiasAdd/ReadVariableOp(^model_1/conv2d_25/Conv2D/ReadVariableOp)^model_1/conv2d_26/BiasAdd/ReadVariableOp(^model_1/conv2d_26/Conv2D/ReadVariableOp)^model_1/conv2d_27/BiasAdd/ReadVariableOp(^model_1/conv2d_27/Conv2D/ReadVariableOp)^model_1/conv2d_28/BiasAdd/ReadVariableOp(^model_1/conv2d_28/Conv2D/ReadVariableOp)^model_1/conv2d_29/BiasAdd/ReadVariableOp(^model_1/conv2d_29/Conv2D/ReadVariableOp)^model_1/conv2d_30/BiasAdd/ReadVariableOp(^model_1/conv2d_30/Conv2D/ReadVariableOp)^model_1/conv2d_31/BiasAdd/ReadVariableOp(^model_1/conv2d_31/Conv2D/ReadVariableOp)^model_1/conv2d_32/BiasAdd/ReadVariableOp(^model_1/conv2d_32/Conv2D/ReadVariableOp)^model_1/conv2d_33/BiasAdd/ReadVariableOp(^model_1/conv2d_33/Conv2D/ReadVariableOp)^model_1/conv2d_34/BiasAdd/ReadVariableOp(^model_1/conv2d_34/Conv2D/ReadVariableOp)^model_1/conv2d_35/BiasAdd/ReadVariableOp(^model_1/conv2d_35/Conv2D/ReadVariableOp)^model_1/conv2d_36/BiasAdd/ReadVariableOp(^model_1/conv2d_36/Conv2D/ReadVariableOp)^model_1/conv2d_37/BiasAdd/ReadVariableOp(^model_1/conv2d_37/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::2~
=model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp=model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
?model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?model_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12\
,model_1/batch_normalization_1/ReadVariableOp,model_1/batch_normalization_1/ReadVariableOp2`
.model_1/batch_normalization_1/ReadVariableOp_1.model_1/batch_normalization_1/ReadVariableOp_12T
(model_1/conv2d_19/BiasAdd/ReadVariableOp(model_1/conv2d_19/BiasAdd/ReadVariableOp2R
'model_1/conv2d_19/Conv2D/ReadVariableOp'model_1/conv2d_19/Conv2D/ReadVariableOp2T
(model_1/conv2d_20/BiasAdd/ReadVariableOp(model_1/conv2d_20/BiasAdd/ReadVariableOp2R
'model_1/conv2d_20/Conv2D/ReadVariableOp'model_1/conv2d_20/Conv2D/ReadVariableOp2T
(model_1/conv2d_21/BiasAdd/ReadVariableOp(model_1/conv2d_21/BiasAdd/ReadVariableOp2R
'model_1/conv2d_21/Conv2D/ReadVariableOp'model_1/conv2d_21/Conv2D/ReadVariableOp2T
(model_1/conv2d_22/BiasAdd/ReadVariableOp(model_1/conv2d_22/BiasAdd/ReadVariableOp2R
'model_1/conv2d_22/Conv2D/ReadVariableOp'model_1/conv2d_22/Conv2D/ReadVariableOp2T
(model_1/conv2d_23/BiasAdd/ReadVariableOp(model_1/conv2d_23/BiasAdd/ReadVariableOp2R
'model_1/conv2d_23/Conv2D/ReadVariableOp'model_1/conv2d_23/Conv2D/ReadVariableOp2T
(model_1/conv2d_24/BiasAdd/ReadVariableOp(model_1/conv2d_24/BiasAdd/ReadVariableOp2R
'model_1/conv2d_24/Conv2D/ReadVariableOp'model_1/conv2d_24/Conv2D/ReadVariableOp2T
(model_1/conv2d_25/BiasAdd/ReadVariableOp(model_1/conv2d_25/BiasAdd/ReadVariableOp2R
'model_1/conv2d_25/Conv2D/ReadVariableOp'model_1/conv2d_25/Conv2D/ReadVariableOp2T
(model_1/conv2d_26/BiasAdd/ReadVariableOp(model_1/conv2d_26/BiasAdd/ReadVariableOp2R
'model_1/conv2d_26/Conv2D/ReadVariableOp'model_1/conv2d_26/Conv2D/ReadVariableOp2T
(model_1/conv2d_27/BiasAdd/ReadVariableOp(model_1/conv2d_27/BiasAdd/ReadVariableOp2R
'model_1/conv2d_27/Conv2D/ReadVariableOp'model_1/conv2d_27/Conv2D/ReadVariableOp2T
(model_1/conv2d_28/BiasAdd/ReadVariableOp(model_1/conv2d_28/BiasAdd/ReadVariableOp2R
'model_1/conv2d_28/Conv2D/ReadVariableOp'model_1/conv2d_28/Conv2D/ReadVariableOp2T
(model_1/conv2d_29/BiasAdd/ReadVariableOp(model_1/conv2d_29/BiasAdd/ReadVariableOp2R
'model_1/conv2d_29/Conv2D/ReadVariableOp'model_1/conv2d_29/Conv2D/ReadVariableOp2T
(model_1/conv2d_30/BiasAdd/ReadVariableOp(model_1/conv2d_30/BiasAdd/ReadVariableOp2R
'model_1/conv2d_30/Conv2D/ReadVariableOp'model_1/conv2d_30/Conv2D/ReadVariableOp2T
(model_1/conv2d_31/BiasAdd/ReadVariableOp(model_1/conv2d_31/BiasAdd/ReadVariableOp2R
'model_1/conv2d_31/Conv2D/ReadVariableOp'model_1/conv2d_31/Conv2D/ReadVariableOp2T
(model_1/conv2d_32/BiasAdd/ReadVariableOp(model_1/conv2d_32/BiasAdd/ReadVariableOp2R
'model_1/conv2d_32/Conv2D/ReadVariableOp'model_1/conv2d_32/Conv2D/ReadVariableOp2T
(model_1/conv2d_33/BiasAdd/ReadVariableOp(model_1/conv2d_33/BiasAdd/ReadVariableOp2R
'model_1/conv2d_33/Conv2D/ReadVariableOp'model_1/conv2d_33/Conv2D/ReadVariableOp2T
(model_1/conv2d_34/BiasAdd/ReadVariableOp(model_1/conv2d_34/BiasAdd/ReadVariableOp2R
'model_1/conv2d_34/Conv2D/ReadVariableOp'model_1/conv2d_34/Conv2D/ReadVariableOp2T
(model_1/conv2d_35/BiasAdd/ReadVariableOp(model_1/conv2d_35/BiasAdd/ReadVariableOp2R
'model_1/conv2d_35/Conv2D/ReadVariableOp'model_1/conv2d_35/Conv2D/ReadVariableOp2T
(model_1/conv2d_36/BiasAdd/ReadVariableOp(model_1/conv2d_36/BiasAdd/ReadVariableOp2R
'model_1/conv2d_36/Conv2D/ReadVariableOp'model_1/conv2d_36/Conv2D/ReadVariableOp2T
(model_1/conv2d_37/BiasAdd/ReadVariableOp(model_1/conv2d_37/BiasAdd/ReadVariableOp2R
'model_1/conv2d_37/Conv2D/ReadVariableOp'model_1/conv2d_37/Conv2D/ReadVariableOp:\ X
1
_output_shapes
:???????????
#
_user_specified_name	RGB_Input
?	
c
G__inference_cropping2d_1_layer_call_and_return_conditional_losses_35902

inputs
identity?
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*J
_output_shapes8
6:4????????????????????????????????????*

begin_mask	*
end_mask2
strided_slice?
IdentityIdentitystrided_slice:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_26_layer_call_and_return_conditional_losses_38154

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
B__inference_model_1_layer_call_and_return_conditional_losses_37075

inputs
batch_normalization_1_36956
batch_normalization_1_36958
batch_normalization_1_36960
batch_normalization_1_36962
conv2d_19_36965
conv2d_19_36967
conv2d_20_36970
conv2d_20_36972
conv2d_21_36976
conv2d_21_36978
conv2d_22_36981
conv2d_22_36983
conv2d_23_36987
conv2d_23_36989
conv2d_24_36992
conv2d_24_36994
conv2d_25_36998
conv2d_25_37000
conv2d_26_37003
conv2d_26_37005
conv2d_27_37009
conv2d_27_37011
conv2d_28_37014
conv2d_28_37016
conv2d_29_37021
conv2d_29_37023
conv2d_30_37026
conv2d_30_37028
conv2d_31_37033
conv2d_31_37035
conv2d_32_37038
conv2d_32_37040
conv2d_33_37045
conv2d_33_37047
conv2d_34_37050
conv2d_34_37052
conv2d_35_37057
conv2d_35_37059
conv2d_36_37062
conv2d_36_37064
conv2d_37_37067
conv2d_37_37069
identity??-batch_normalization_1/StatefulPartitionedCall?!conv2d_19/StatefulPartitionedCall?!conv2d_20/StatefulPartitionedCall?!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?!conv2d_24/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?!conv2d_30/StatefulPartitionedCall?!conv2d_31/StatefulPartitionedCall?!conv2d_32/StatefulPartitionedCall?!conv2d_33/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall?!conv2d_35/StatefulPartitionedCall?!conv2d_36/StatefulPartitionedCall?!conv2d_37/StatefulPartitionedCall?
 gaussian_noise_1/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_359402"
 gaussian_noise_1/PartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)gaussian_noise_1/PartitionedCall:output:0batch_normalization_1_36956batch_normalization_1_36958batch_normalization_1_36960batch_normalization_1_36962*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_359902/
-batch_normalization_1/StatefulPartitionedCall?
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv2d_19_36965conv2d_19_36967*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_360372#
!conv2d_19/StatefulPartitionedCall?
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0conv2d_20_36970conv2d_20_36972*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_360642#
!conv2d_20/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_357752!
max_pooling2d_4/PartitionedCall?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_21_36976conv2d_21_36978*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_360922#
!conv2d_21/StatefulPartitionedCall?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_36981conv2d_22_36983*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_361192#
!conv2d_22/StatefulPartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_357872!
max_pooling2d_5/PartitionedCall?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_23_36987conv2d_23_36989*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_361472#
!conv2d_23/StatefulPartitionedCall?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0conv2d_24_36992conv2d_24_36994*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_24_layer_call_and_return_conditional_losses_361742#
!conv2d_24/StatefulPartitionedCall?
max_pooling2d_6/PartitionedCallPartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_357992!
max_pooling2d_6/PartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:0conv2d_25_36998conv2d_25_37000*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_362022#
!conv2d_25/StatefulPartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0conv2d_26_37003conv2d_26_37005*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_26_layer_call_and_return_conditional_losses_362292#
!conv2d_26/StatefulPartitionedCall?
max_pooling2d_7/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_358112!
max_pooling2d_7/PartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:0conv2d_27_37009conv2d_27_37011*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_362572#
!conv2d_27/StatefulPartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_37014conv2d_28_37016*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_28_layer_call_and_return_conditional_losses_362842#
!conv2d_28/StatefulPartitionedCall?
up_sampling2d_4/PartitionedCallPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_358302!
up_sampling2d_4/PartitionedCall?
concatenate_4/PartitionedCallPartitionedCall(up_sampling2d_4/PartitionedCall:output:0*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_4_layer_call_and_return_conditional_losses_363082
concatenate_4/PartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall&concatenate_4/PartitionedCall:output:0conv2d_29_37021conv2d_29_37023*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_29_layer_call_and_return_conditional_losses_363282#
!conv2d_29/StatefulPartitionedCall?
!conv2d_30/StatefulPartitionedCallStatefulPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0conv2d_30_37026conv2d_30_37028*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_30_layer_call_and_return_conditional_losses_363552#
!conv2d_30/StatefulPartitionedCall?
up_sampling2d_5/PartitionedCallPartitionedCall*conv2d_30/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_358492!
up_sampling2d_5/PartitionedCall?
concatenate_5/PartitionedCallPartitionedCall(up_sampling2d_5/PartitionedCall:output:0*conv2d_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_5_layer_call_and_return_conditional_losses_363792
concatenate_5/PartitionedCall?
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCall&concatenate_5/PartitionedCall:output:0conv2d_31_37033conv2d_31_37035*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_31_layer_call_and_return_conditional_losses_363992#
!conv2d_31/StatefulPartitionedCall?
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0conv2d_32_37038conv2d_32_37040*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_32_layer_call_and_return_conditional_losses_364262#
!conv2d_32/StatefulPartitionedCall?
up_sampling2d_6/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_358682!
up_sampling2d_6/PartitionedCall?
concatenate_6/PartitionedCallPartitionedCall(up_sampling2d_6/PartitionedCall:output:0*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_6_layer_call_and_return_conditional_losses_364502
concatenate_6/PartitionedCall?
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall&concatenate_6/PartitionedCall:output:0conv2d_33_37045conv2d_33_37047*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_33_layer_call_and_return_conditional_losses_364702#
!conv2d_33/StatefulPartitionedCall?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0conv2d_34_37050conv2d_34_37052*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_34_layer_call_and_return_conditional_losses_364972#
!conv2d_34/StatefulPartitionedCall?
up_sampling2d_7/PartitionedCallPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_358872!
up_sampling2d_7/PartitionedCall?
concatenate_7/PartitionedCallPartitionedCall(up_sampling2d_7/PartitionedCall:output:0*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_7_layer_call_and_return_conditional_losses_365212
concatenate_7/PartitionedCall?
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0conv2d_35_37057conv2d_35_37059*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_35_layer_call_and_return_conditional_losses_365412#
!conv2d_35/StatefulPartitionedCall?
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0conv2d_36_37062conv2d_36_37064*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_36_layer_call_and_return_conditional_losses_365682#
!conv2d_36/StatefulPartitionedCall?
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_37067conv2d_37_37069*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_37_layer_call_and_return_conditional_losses_365952#
!conv2d_37/StatefulPartitionedCall?
cropping2d_1/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_cropping2d_1_layer_call_and_return_conditional_losses_359022
cropping2d_1/PartitionedCall?
 zero_padding2d_1/PartitionedCallPartitionedCall%cropping2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_359152"
 zero_padding2d_1/PartitionedCall?
IdentityIdentity)zero_padding2d_1/PartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall"^conv2d_30/StatefulPartitionedCall"^conv2d_31/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:???????????::::::::::::::::::::::::::::::::::::::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2F
!conv2d_30/StatefulPartitionedCall!conv2d_30/StatefulPartitionedCall2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
I
	RGB_Input<
serving_default_RGB_Input:0???????????N
zero_padding2d_1:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
??
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer-14
layer_with_weights-9
layer-15
layer_with_weights-10
layer-16
layer-17
layer-18
layer_with_weights-11
layer-19
layer_with_weights-12
layer-20
layer-21
layer-22
layer_with_weights-13
layer-23
layer_with_weights-14
layer-24
layer-25
layer-26
layer_with_weights-15
layer-27
layer_with_weights-16
layer-28
layer-29
layer-30
 layer_with_weights-17
 layer-31
!layer_with_weights-18
!layer-32
"layer_with_weights-19
"layer-33
#layer-34
$layer-35
%	optimizer
&regularization_losses
'	variables
(trainable_variables
)	keras_api
*
signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"»
_tf_keras_network??{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "RGB_Input"}, "name": "RGB_Input", "inbound_nodes": []}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_1", "trainable": true, "dtype": "float32", "stddev": 0.1}, "name": "gaussian_noise_1", "inbound_nodes": [[["RGB_Input", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["gaussian_noise_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_19", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_20", "inbound_nodes": [[["conv2d_19", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["conv2d_20", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_21", "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_22", "inbound_nodes": [[["conv2d_21", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_5", "inbound_nodes": [[["conv2d_22", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_23", "inbound_nodes": [[["max_pooling2d_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_24", "inbound_nodes": [[["conv2d_23", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_6", "inbound_nodes": [[["conv2d_24", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_25", "inbound_nodes": [[["max_pooling2d_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_26", "inbound_nodes": [[["conv2d_25", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_7", "inbound_nodes": [[["conv2d_26", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["max_pooling2d_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_28", "inbound_nodes": [[["conv2d_27", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_4", "inbound_nodes": [[["conv2d_28", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_4", "inbound_nodes": [[["up_sampling2d_4", 0, 0, {}], ["conv2d_26", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_29", "inbound_nodes": [[["concatenate_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_30", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_30", "inbound_nodes": [[["conv2d_29", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_5", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_5", "inbound_nodes": [[["conv2d_30", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_5", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_5", "inbound_nodes": [[["up_sampling2d_5", 0, 0, {}], ["conv2d_24", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_31", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_31", "inbound_nodes": [[["concatenate_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_32", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_32", "inbound_nodes": [[["conv2d_31", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_6", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_6", "inbound_nodes": [[["conv2d_32", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_6", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_6", "inbound_nodes": [[["up_sampling2d_6", 0, 0, {}], ["conv2d_22", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_33", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_33", "inbound_nodes": [[["concatenate_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_34", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_34", "inbound_nodes": [[["conv2d_33", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_7", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_7", "inbound_nodes": [[["conv2d_34", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_7", "trainable": true, "dtype": "float32", "axis": 3}, "name": "concatenate_7", "inbound_nodes": [[["up_sampling2d_7", 0, 0, {}], ["conv2d_20", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_35", "inbound_nodes": [[["concatenate_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_36", "inbound_nodes": [[["conv2d_35", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_37", "inbound_nodes": [[["conv2d_36", 0, 0, {}]]]}, {"class_name": "Cropping2D", "config": {"name": "cropping2d_1", "trainable": true, "dtype": "float32", "cropping": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 0]}]}, "data_format": "channels_last"}, "name": "cropping2d_1", "inbound_nodes": [[["conv2d_37", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 0]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_1", "inbound_nodes": [[["cropping2d_1", 0, 0, {}]]]}], "input_layers": [["RGB_Input", 0, 0]], "output_layers": [["zero_padding2d_1", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "RGB_Input"}, "name": "RGB_Input", "inbound_nodes": []}, {"class_name": "GaussianNoise", "config": {"name": "gaussian_noise_1", "trainable": true, "dtype": "float32", "stddev": 0.1}, "name": "gaussian_noise_1", "inbound_nodes": [[["RGB_Input", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["gaussian_noise_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_19", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_20", "inbound_nodes": [[["conv2d_19", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["conv2d_20", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_21", "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_22", "inbound_nodes": [[["conv2d_21", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_5", "inbound_nodes": [[["conv2d_22", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_23", "inbound_nodes": [[["max_pooling2d_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_24", "inbound_nodes": [[["conv2d_23", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_6", "inbound_nodes": [[["conv2d_24", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_25", "inbound_nodes": [[["max_pooling2d_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_26", "inbound_nodes": [[["conv2d_25", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_7", "inbound_nodes": [[["conv2d_26", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["max_pooling2d_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_28", "inbound_nodes": [[["conv2d_27", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_4", "inbound_nodes": [[["conv2d_28", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_4", "inbound_nodes": [[["up_sampling2d_4", 0, 0, {}], ["conv2d_26", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_29", "inbound_nodes": [[["concatenate_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_30", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_30", "inbound_nodes": [[["conv2d_29", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_5", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_5", "inbound_nodes": [[["conv2d_30", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_5", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_5", "inbound_nodes": [[["up_sampling2d_5", 0, 0, {}], ["conv2d_24", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_31", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_31", "inbound_nodes": [[["concatenate_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_32", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_32", "inbound_nodes": [[["conv2d_31", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_6", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_6", "inbound_nodes": [[["conv2d_32", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_6", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_6", "inbound_nodes": [[["up_sampling2d_6", 0, 0, {}], ["conv2d_22", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_33", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_33", "inbound_nodes": [[["concatenate_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_34", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_34", "inbound_nodes": [[["conv2d_33", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_7", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_7", "inbound_nodes": [[["conv2d_34", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_7", "trainable": true, "dtype": "float32", "axis": 3}, "name": "concatenate_7", "inbound_nodes": [[["up_sampling2d_7", 0, 0, {}], ["conv2d_20", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_35", "inbound_nodes": [[["concatenate_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_36", "inbound_nodes": [[["conv2d_35", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_37", "inbound_nodes": [[["conv2d_36", 0, 0, {}]]]}, {"class_name": "Cropping2D", "config": {"name": "cropping2d_1", "trainable": true, "dtype": "float32", "cropping": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 0]}]}, "data_format": "channels_last"}, "name": "cropping2d_1", "inbound_nodes": [[["conv2d_37", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 0]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_1", "inbound_nodes": [[["cropping2d_1", 0, 0, {}]]]}], "input_layers": [["RGB_Input", 0, 0]], "output_layers": [["zero_padding2d_1", 0, 0]]}}, "training_config": {"loss": "dice_p_bce", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "dice_coef", "dtype": "float32", "fn": "dice_coef"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "binary_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "true_positive_rate", "dtype": "float32", "fn": "true_positive_rate"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 9.999999974752427e-07, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "RGB_Input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "RGB_Input"}}
?
+	variables
,regularization_losses
-trainable_variables
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "GaussianNoise", "name": "gaussian_noise_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_noise_1", "trainable": true, "dtype": "float32", "stddev": 0.1}}
?	
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5regularization_losses
6trainable_variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 3]}}
?	

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_19", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 3]}}
?	

>kernel
?bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_20", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?
D	variables
Eregularization_losses
Ftrainable_variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

Hkernel
Ibias
J	variables
Kregularization_losses
Ltrainable_variables
M	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 8]}}
?	

Nkernel
Obias
P	variables
Qregularization_losses
Rtrainable_variables
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 16]}}
?
T	variables
Uregularization_losses
Vtrainable_variables
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

Xkernel
Ybias
Z	variables
[regularization_losses
\trainable_variables
]	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
?	

^kernel
_bias
`	variables
aregularization_losses
btrainable_variables
c	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

hkernel
ibias
j	variables
kregularization_losses
ltrainable_variables
m	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}
?	

nkernel
obias
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
?
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

xkernel
ybias
z	variables
{regularization_losses
|trainable_variables
}	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
?	

~kernel
bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 128]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_4", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 16, 16, 128]}, {"class_name": "TensorShape", "items": [null, 16, 16, 64]}]}
?	
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 192}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 192]}}
?	
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_30", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_30", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_5", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_5", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 32, 64]}, {"class_name": "TensorShape", "items": [null, 32, 32, 32]}]}
?	
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_31", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 96}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 96]}}
?	
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_32", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 32]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_6", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_6", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 64, 64, 32]}, {"class_name": "TensorShape", "items": [null, 64, 64, 16]}]}
?	
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_33", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 48}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 48]}}
?	
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_34", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 16]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_7", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_7", "trainable": true, "dtype": "float32", "axis": 3}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128, 128, 16]}, {"class_name": "TensorShape", "items": [null, 128, 128, 8]}]}
?	
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_35", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 24}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 24]}}
?	
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_36", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?	
?kernel
	?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_37", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Cropping2D", "name": "cropping2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "cropping2d_1", "trainable": true, "dtype": "float32", "cropping": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 0]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "ZeroPadding2D", "name": "zero_padding2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d_1", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [0, 0]}, {"class_name": "__tuple__", "items": [0, 0]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
	?iter
?beta_1
?beta_2

?decay
?learning_rate0m?1m?8m?9m?>m??m?Hm?Im?Nm?Om?Xm?Ym?^m?_m?hm?im?nm?om?xm?ym?~m?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?0v?1v?8v?9v?>v??v?Hv?Iv?Nv?Ov?Xv?Yv?^v?_v?hv?iv?nv?ov?xv?yv?~v?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
 "
trackable_list_wrapper
?
00
11
22
33
84
95
>6
?7
H8
I9
N10
O11
X12
Y13
^14
_15
h16
i17
n18
o19
x20
y21
~22
23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41"
trackable_list_wrapper
?
00
11
82
93
>4
?5
H6
I7
N8
O9
X10
Y11
^12
_13
h14
i15
n16
o17
x18
y19
~20
21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39"
trackable_list_wrapper
?
?metrics
?layer_metrics
?layers
 ?layer_regularization_losses
&regularization_losses
'	variables
(trainable_variables
?non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
+	variables
?layer_metrics
?layers
 ?layer_regularization_losses
,regularization_losses
-trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
<
00
11
22
33"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
?metrics
4	variables
?layer_metrics
?layers
 ?layer_regularization_losses
5regularization_losses
6trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_19/kernel
:2conv2d_19/bias
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
?
?metrics
:	variables
?layer_metrics
?layers
 ?layer_regularization_losses
;regularization_losses
<trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_20/kernel
:2conv2d_20/bias
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
?
?metrics
@	variables
?layer_metrics
?layers
 ?layer_regularization_losses
Aregularization_losses
Btrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
D	variables
?layer_metrics
?layers
 ?layer_regularization_losses
Eregularization_losses
Ftrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_21/kernel
:2conv2d_21/bias
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
?
?metrics
J	variables
?layer_metrics
?layers
 ?layer_regularization_losses
Kregularization_losses
Ltrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_22/kernel
:2conv2d_22/bias
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
?
?metrics
P	variables
?layer_metrics
?layers
 ?layer_regularization_losses
Qregularization_losses
Rtrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
T	variables
?layer_metrics
?layers
 ?layer_regularization_losses
Uregularization_losses
Vtrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_23/kernel
: 2conv2d_23/bias
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
?
?metrics
Z	variables
?layer_metrics
?layers
 ?layer_regularization_losses
[regularization_losses
\trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_24/kernel
: 2conv2d_24/bias
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
?
?metrics
`	variables
?layer_metrics
?layers
 ?layer_regularization_losses
aregularization_losses
btrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
d	variables
?layer_metrics
?layers
 ?layer_regularization_losses
eregularization_losses
ftrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_25/kernel
:@2conv2d_25/bias
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
?
?metrics
j	variables
?layer_metrics
?layers
 ?layer_regularization_losses
kregularization_losses
ltrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_26/kernel
:@2conv2d_26/bias
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
?
?metrics
p	variables
?layer_metrics
?layers
 ?layer_regularization_losses
qregularization_losses
rtrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
t	variables
?layer_metrics
?layers
 ?layer_regularization_losses
uregularization_losses
vtrainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)@?2conv2d_27/kernel
:?2conv2d_27/bias
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
?
?metrics
z	variables
?layer_metrics
?layers
 ?layer_regularization_losses
{regularization_losses
|trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_28/kernel
:?2conv2d_28/bias
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)?@2conv2d_29/kernel
:@2conv2d_29/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_30/kernel
:@2conv2d_30/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(` 2conv2d_31/kernel
: 2conv2d_31/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_32/kernel
: 2conv2d_32/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(02conv2d_33/kernel
:2conv2d_33/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_34/kernel
:2conv2d_34/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_35/kernel
:2conv2d_35/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_36/kernel
:2conv2d_36/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_37/kernel
:2conv2d_37/bias
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?	variables
?layer_metrics
?layers
 ?layer_regularization_losses
?regularization_losses
?trainable_variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_dict_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35"
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "dice_coef", "dtype": "float32", "config": {"name": "dice_coef", "dtype": "float32", "fn": "dice_coef"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "binary_accuracy", "dtype": "float32", "config": {"name": "binary_accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "true_positive_rate", "dtype": "float32", "config": {"name": "true_positive_rate", "dtype": "float32", "fn": "true_positive_rate"}}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
.:,2"Adam/batch_normalization_1/gamma/m
-:+2!Adam/batch_normalization_1/beta/m
/:-2Adam/conv2d_19/kernel/m
!:2Adam/conv2d_19/bias/m
/:-2Adam/conv2d_20/kernel/m
!:2Adam/conv2d_20/bias/m
/:-2Adam/conv2d_21/kernel/m
!:2Adam/conv2d_21/bias/m
/:-2Adam/conv2d_22/kernel/m
!:2Adam/conv2d_22/bias/m
/:- 2Adam/conv2d_23/kernel/m
!: 2Adam/conv2d_23/bias/m
/:-  2Adam/conv2d_24/kernel/m
!: 2Adam/conv2d_24/bias/m
/:- @2Adam/conv2d_25/kernel/m
!:@2Adam/conv2d_25/bias/m
/:-@@2Adam/conv2d_26/kernel/m
!:@2Adam/conv2d_26/bias/m
0:.@?2Adam/conv2d_27/kernel/m
": ?2Adam/conv2d_27/bias/m
1:/??2Adam/conv2d_28/kernel/m
": ?2Adam/conv2d_28/bias/m
0:.?@2Adam/conv2d_29/kernel/m
!:@2Adam/conv2d_29/bias/m
/:-@@2Adam/conv2d_30/kernel/m
!:@2Adam/conv2d_30/bias/m
/:-` 2Adam/conv2d_31/kernel/m
!: 2Adam/conv2d_31/bias/m
/:-  2Adam/conv2d_32/kernel/m
!: 2Adam/conv2d_32/bias/m
/:-02Adam/conv2d_33/kernel/m
!:2Adam/conv2d_33/bias/m
/:-2Adam/conv2d_34/kernel/m
!:2Adam/conv2d_34/bias/m
/:-2Adam/conv2d_35/kernel/m
!:2Adam/conv2d_35/bias/m
/:-2Adam/conv2d_36/kernel/m
!:2Adam/conv2d_36/bias/m
/:-2Adam/conv2d_37/kernel/m
!:2Adam/conv2d_37/bias/m
.:,2"Adam/batch_normalization_1/gamma/v
-:+2!Adam/batch_normalization_1/beta/v
/:-2Adam/conv2d_19/kernel/v
!:2Adam/conv2d_19/bias/v
/:-2Adam/conv2d_20/kernel/v
!:2Adam/conv2d_20/bias/v
/:-2Adam/conv2d_21/kernel/v
!:2Adam/conv2d_21/bias/v
/:-2Adam/conv2d_22/kernel/v
!:2Adam/conv2d_22/bias/v
/:- 2Adam/conv2d_23/kernel/v
!: 2Adam/conv2d_23/bias/v
/:-  2Adam/conv2d_24/kernel/v
!: 2Adam/conv2d_24/bias/v
/:- @2Adam/conv2d_25/kernel/v
!:@2Adam/conv2d_25/bias/v
/:-@@2Adam/conv2d_26/kernel/v
!:@2Adam/conv2d_26/bias/v
0:.@?2Adam/conv2d_27/kernel/v
": ?2Adam/conv2d_27/bias/v
1:/??2Adam/conv2d_28/kernel/v
": ?2Adam/conv2d_28/bias/v
0:.?@2Adam/conv2d_29/kernel/v
!:@2Adam/conv2d_29/bias/v
/:-@@2Adam/conv2d_30/kernel/v
!:@2Adam/conv2d_30/bias/v
/:-` 2Adam/conv2d_31/kernel/v
!: 2Adam/conv2d_31/bias/v
/:-  2Adam/conv2d_32/kernel/v
!: 2Adam/conv2d_32/bias/v
/:-02Adam/conv2d_33/kernel/v
!:2Adam/conv2d_33/bias/v
/:-2Adam/conv2d_34/kernel/v
!:2Adam/conv2d_34/bias/v
/:-2Adam/conv2d_35/kernel/v
!:2Adam/conv2d_35/bias/v
/:-2Adam/conv2d_36/kernel/v
!:2Adam/conv2d_36/bias/v
/:-2Adam/conv2d_37/kernel/v
!:2Adam/conv2d_37/bias/v
?2?
 __inference__wrapped_model_35665?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *2?/
-?*
	RGB_Input???????????
?2?
B__inference_model_1_layer_call_and_return_conditional_losses_37471
B__inference_model_1_layer_call_and_return_conditional_losses_36737
B__inference_model_1_layer_call_and_return_conditional_losses_37672
B__inference_model_1_layer_call_and_return_conditional_losses_36614?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_model_1_layer_call_fn_37761
'__inference_model_1_layer_call_fn_36950
'__inference_model_1_layer_call_fn_37162
'__inference_model_1_layer_call_fn_37850?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_37861
K__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_37865?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_gaussian_noise_1_layer_call_fn_37875
0__inference_gaussian_noise_1_layer_call_fn_37870?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_37959
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_37977
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_37895
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_37913?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
5__inference_batch_normalization_1_layer_call_fn_37939
5__inference_batch_normalization_1_layer_call_fn_37926
5__inference_batch_normalization_1_layer_call_fn_37990
5__inference_batch_normalization_1_layer_call_fn_38003?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_conv2d_19_layer_call_and_return_conditional_losses_38014?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_19_layer_call_fn_38023?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_20_layer_call_and_return_conditional_losses_38034?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_20_layer_call_fn_38043?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_35775?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_max_pooling2d_4_layer_call_fn_35781?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_conv2d_21_layer_call_and_return_conditional_losses_38054?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_21_layer_call_fn_38063?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_22_layer_call_and_return_conditional_losses_38074?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_22_layer_call_fn_38083?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_35787?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_max_pooling2d_5_layer_call_fn_35793?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_conv2d_23_layer_call_and_return_conditional_losses_38094?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_23_layer_call_fn_38103?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_24_layer_call_and_return_conditional_losses_38114?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_24_layer_call_fn_38123?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_35799?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_max_pooling2d_6_layer_call_fn_35805?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_conv2d_25_layer_call_and_return_conditional_losses_38134?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_25_layer_call_fn_38143?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_26_layer_call_and_return_conditional_losses_38154?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_26_layer_call_fn_38163?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_35811?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_max_pooling2d_7_layer_call_fn_35817?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
D__inference_conv2d_27_layer_call_and_return_conditional_losses_38174?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_27_layer_call_fn_38183?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_28_layer_call_and_return_conditional_losses_38194?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_28_layer_call_fn_38203?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_35830?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_up_sampling2d_4_layer_call_fn_35836?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
H__inference_concatenate_4_layer_call_and_return_conditional_losses_38210?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_concatenate_4_layer_call_fn_38216?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_29_layer_call_and_return_conditional_losses_38227?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_29_layer_call_fn_38236?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_30_layer_call_and_return_conditional_losses_38247?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_30_layer_call_fn_38256?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_35849?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_up_sampling2d_5_layer_call_fn_35855?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
H__inference_concatenate_5_layer_call_and_return_conditional_losses_38263?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_concatenate_5_layer_call_fn_38269?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_31_layer_call_and_return_conditional_losses_38280?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_31_layer_call_fn_38289?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_32_layer_call_and_return_conditional_losses_38300?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_32_layer_call_fn_38309?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_35868?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_up_sampling2d_6_layer_call_fn_35874?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
H__inference_concatenate_6_layer_call_and_return_conditional_losses_38316?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_concatenate_6_layer_call_fn_38322?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_33_layer_call_and_return_conditional_losses_38333?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_33_layer_call_fn_38342?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_34_layer_call_and_return_conditional_losses_38353?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_34_layer_call_fn_38362?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_35887?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
/__inference_up_sampling2d_7_layer_call_fn_35893?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
H__inference_concatenate_7_layer_call_and_return_conditional_losses_38369?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_concatenate_7_layer_call_fn_38375?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_35_layer_call_and_return_conditional_losses_38386?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_35_layer_call_fn_38395?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_36_layer_call_and_return_conditional_losses_38406?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_36_layer_call_fn_38415?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_37_layer_call_and_return_conditional_losses_38426?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_37_layer_call_fn_38435?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_cropping2d_1_layer_call_and_return_conditional_losses_35902?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
,__inference_cropping2d_1_layer_call_fn_35908?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_35915?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
0__inference_zero_padding2d_1_layer_call_fn_35921?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?B?
#__inference_signature_wrapper_37261	RGB_Input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_35665?<012389>?HINOXY^_hinoxy~??????????????????<?9
2?/
-?*
	RGB_Input???????????
? "M?J
H
zero_padding2d_14?1
zero_padding2d_1????????????
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_37895v0123=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_37913v0123=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_37959?0123M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_37977?0123M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
5__inference_batch_normalization_1_layer_call_fn_37926i0123=?:
3?0
*?'
inputs???????????
p
? ""?????????????
5__inference_batch_normalization_1_layer_call_fn_37939i0123=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
5__inference_batch_normalization_1_layer_call_fn_37990?0123M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
5__inference_batch_normalization_1_layer_call_fn_38003?0123M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
H__inference_concatenate_4_layer_call_and_return_conditional_losses_38210?}?z
s?p
n?k
=?:
inputs/0,????????????????????????????
*?'
inputs/1?????????@
? ".?+
$?!
0??????????
? ?
-__inference_concatenate_4_layer_call_fn_38216?}?z
s?p
n?k
=?:
inputs/0,????????????????????????????
*?'
inputs/1?????????@
? "!????????????
H__inference_concatenate_5_layer_call_and_return_conditional_losses_38263?|?y
r?o
m?j
<?9
inputs/0+???????????????????????????@
*?'
inputs/1?????????   
? "-?*
#? 
0?????????  `
? ?
-__inference_concatenate_5_layer_call_fn_38269?|?y
r?o
m?j
<?9
inputs/0+???????????????????????????@
*?'
inputs/1?????????   
? " ??????????  `?
H__inference_concatenate_6_layer_call_and_return_conditional_losses_38316?|?y
r?o
m?j
<?9
inputs/0+??????????????????????????? 
*?'
inputs/1?????????@@
? "-?*
#? 
0?????????@@0
? ?
-__inference_concatenate_6_layer_call_fn_38322?|?y
r?o
m?j
<?9
inputs/0+??????????????????????????? 
*?'
inputs/1?????????@@
? " ??????????@@0?
H__inference_concatenate_7_layer_call_and_return_conditional_losses_38369?~?{
t?q
o?l
<?9
inputs/0+???????????????????????????
,?)
inputs/1???????????
? "/?,
%?"
0???????????
? ?
-__inference_concatenate_7_layer_call_fn_38375?~?{
t?q
o?l
<?9
inputs/0+???????????????????????????
,?)
inputs/1???????????
? ""?????????????
D__inference_conv2d_19_layer_call_and_return_conditional_losses_38014p899?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
)__inference_conv2d_19_layer_call_fn_38023c899?6
/?,
*?'
inputs???????????
? ""?????????????
D__inference_conv2d_20_layer_call_and_return_conditional_losses_38034p>?9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
)__inference_conv2d_20_layer_call_fn_38043c>?9?6
/?,
*?'
inputs???????????
? ""?????????????
D__inference_conv2d_21_layer_call_and_return_conditional_losses_38054lHI7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@
? ?
)__inference_conv2d_21_layer_call_fn_38063_HI7?4
-?*
(?%
inputs?????????@@
? " ??????????@@?
D__inference_conv2d_22_layer_call_and_return_conditional_losses_38074lNO7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@
? ?
)__inference_conv2d_22_layer_call_fn_38083_NO7?4
-?*
(?%
inputs?????????@@
? " ??????????@@?
D__inference_conv2d_23_layer_call_and_return_conditional_losses_38094lXY7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????   
? ?
)__inference_conv2d_23_layer_call_fn_38103_XY7?4
-?*
(?%
inputs?????????  
? " ??????????   ?
D__inference_conv2d_24_layer_call_and_return_conditional_losses_38114l^_7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
)__inference_conv2d_24_layer_call_fn_38123_^_7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
D__inference_conv2d_25_layer_call_and_return_conditional_losses_38134lhi7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_25_layer_call_fn_38143_hi7?4
-?*
(?%
inputs????????? 
? " ??????????@?
D__inference_conv2d_26_layer_call_and_return_conditional_losses_38154lno7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_26_layer_call_fn_38163_no7?4
-?*
(?%
inputs?????????@
? " ??????????@?
D__inference_conv2d_27_layer_call_and_return_conditional_losses_38174mxy7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_27_layer_call_fn_38183`xy7?4
-?*
(?%
inputs?????????@
? "!????????????
D__inference_conv2d_28_layer_call_and_return_conditional_losses_38194n~8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_28_layer_call_fn_38203a~8?5
.?+
)?&
inputs??????????
? "!????????????
D__inference_conv2d_29_layer_call_and_return_conditional_losses_38227o??8?5
.?+
)?&
inputs??????????
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_29_layer_call_fn_38236b??8?5
.?+
)?&
inputs??????????
? " ??????????@?
D__inference_conv2d_30_layer_call_and_return_conditional_losses_38247n??7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_30_layer_call_fn_38256a??7?4
-?*
(?%
inputs?????????@
? " ??????????@?
D__inference_conv2d_31_layer_call_and_return_conditional_losses_38280n??7?4
-?*
(?%
inputs?????????  `
? "-?*
#? 
0?????????   
? ?
)__inference_conv2d_31_layer_call_fn_38289a??7?4
-?*
(?%
inputs?????????  `
? " ??????????   ?
D__inference_conv2d_32_layer_call_and_return_conditional_losses_38300n??7?4
-?*
(?%
inputs?????????   
? "-?*
#? 
0?????????   
? ?
)__inference_conv2d_32_layer_call_fn_38309a??7?4
-?*
(?%
inputs?????????   
? " ??????????   ?
D__inference_conv2d_33_layer_call_and_return_conditional_losses_38333n??7?4
-?*
(?%
inputs?????????@@0
? "-?*
#? 
0?????????@@
? ?
)__inference_conv2d_33_layer_call_fn_38342a??7?4
-?*
(?%
inputs?????????@@0
? " ??????????@@?
D__inference_conv2d_34_layer_call_and_return_conditional_losses_38353n??7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????@@
? ?
)__inference_conv2d_34_layer_call_fn_38362a??7?4
-?*
(?%
inputs?????????@@
? " ??????????@@?
D__inference_conv2d_35_layer_call_and_return_conditional_losses_38386r??9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
)__inference_conv2d_35_layer_call_fn_38395e??9?6
/?,
*?'
inputs???????????
? ""?????????????
D__inference_conv2d_36_layer_call_and_return_conditional_losses_38406r??9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
)__inference_conv2d_36_layer_call_fn_38415e??9?6
/?,
*?'
inputs???????????
? ""?????????????
D__inference_conv2d_37_layer_call_and_return_conditional_losses_38426r??9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
)__inference_conv2d_37_layer_call_fn_38435e??9?6
/?,
*?'
inputs???????????
? ""?????????????
G__inference_cropping2d_1_layer_call_and_return_conditional_losses_35902?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_cropping2d_1_layer_call_fn_35908?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_37861p=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
K__inference_gaussian_noise_1_layer_call_and_return_conditional_losses_37865p=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
0__inference_gaussian_noise_1_layer_call_fn_37870c=?:
3?0
*?'
inputs???????????
p
? ""?????????????
0__inference_gaussian_noise_1_layer_call_fn_37875c=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_35775?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_4_layer_call_fn_35781?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_35787?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_5_layer_call_fn_35793?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_35799?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_6_layer_call_fn_35805?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_35811?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_7_layer_call_fn_35817?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_model_1_layer_call_and_return_conditional_losses_36614?<012389>?HINOXY^_hinoxy~??????????????????D?A
:?7
-?*
	RGB_Input???????????
p

 
? "/?,
%?"
0???????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_36737?<012389>?HINOXY^_hinoxy~??????????????????D?A
:?7
-?*
	RGB_Input???????????
p 

 
? "/?,
%?"
0???????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_37471?<012389>?HINOXY^_hinoxy~??????????????????A?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
B__inference_model_1_layer_call_and_return_conditional_losses_37672?<012389>?HINOXY^_hinoxy~??????????????????A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
'__inference_model_1_layer_call_fn_36950?<012389>?HINOXY^_hinoxy~??????????????????D?A
:?7
-?*
	RGB_Input???????????
p

 
? ""?????????????
'__inference_model_1_layer_call_fn_37162?<012389>?HINOXY^_hinoxy~??????????????????D?A
:?7
-?*
	RGB_Input???????????
p 

 
? ""?????????????
'__inference_model_1_layer_call_fn_37761?<012389>?HINOXY^_hinoxy~??????????????????A?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
'__inference_model_1_layer_call_fn_37850?<012389>?HINOXY^_hinoxy~??????????????????A?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
#__inference_signature_wrapper_37261?<012389>?HINOXY^_hinoxy~??????????????????I?F
? 
??<
:
	RGB_Input-?*
	RGB_Input???????????"M?J
H
zero_padding2d_14?1
zero_padding2d_1????????????
J__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_35830?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_4_layer_call_fn_35836?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_up_sampling2d_5_layer_call_and_return_conditional_losses_35849?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_5_layer_call_fn_35855?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_35868?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_6_layer_call_fn_35874?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_35887?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_7_layer_call_fn_35893?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_zero_padding2d_1_layer_call_and_return_conditional_losses_35915?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_zero_padding2d_1_layer_call_fn_35921?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????