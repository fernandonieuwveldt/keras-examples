Ç	
·
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
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
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
$

LogicalAnd
x

y

z

q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.7.02unknown8ä
x
batch_norm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatch_norm/gamma
q
$batch_norm/gamma/Read/ReadVariableOpReadVariableOpbatch_norm/gamma*
_output_shapes
:*
dtype0
v
batch_norm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namebatch_norm/beta
o
#batch_norm/beta/Read/ReadVariableOpReadVariableOpbatch_norm/beta*
_output_shapes
:*
dtype0

batch_norm/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namebatch_norm/moving_mean
}
*batch_norm/moving_mean/Read/ReadVariableOpReadVariableOpbatch_norm/moving_mean*
_output_shapes
:*
dtype0

batch_norm/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_norm/moving_variance

.batch_norm/moving_variance/Read/ReadVariableOpReadVariableOpbatch_norm/moving_variance*
_output_shapes
:*
dtype0
v
target/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nametarget/kernel
o
!target/kernel/Read/ReadVariableOpReadVariableOptarget/kernel*
_output_shapes

:*
dtype0
n
target/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametarget/bias
g
target/bias/Read/ReadVariableOpReadVariableOptarget/bias*
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
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:È*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:È*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:È*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:È*
dtype0

Adam/batch_norm/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/batch_norm/gamma/m

+Adam/batch_norm/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batch_norm/gamma/m*
_output_shapes
:*
dtype0

Adam/batch_norm/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/batch_norm/beta/m
}
*Adam/batch_norm/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_norm/beta/m*
_output_shapes
:*
dtype0

Adam/target/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/target/kernel/m
}
(Adam/target/kernel/m/Read/ReadVariableOpReadVariableOpAdam/target/kernel/m*
_output_shapes

:*
dtype0
|
Adam/target/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/target/bias/m
u
&Adam/target/bias/m/Read/ReadVariableOpReadVariableOpAdam/target/bias/m*
_output_shapes
:*
dtype0

Adam/batch_norm/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/batch_norm/gamma/v

+Adam/batch_norm/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batch_norm/gamma/v*
_output_shapes
:*
dtype0

Adam/batch_norm/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/batch_norm/beta/v
}
*Adam/batch_norm/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_norm/beta/v*
_output_shapes
:*
dtype0

Adam/target/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/target/kernel/v
}
(Adam/target/kernel/v/Read/ReadVariableOpReadVariableOpAdam/target/kernel/v*
_output_shapes

:*
dtype0
|
Adam/target/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/target/bias/v
u
&Adam/target/bias/v/Read/ReadVariableOpReadVariableOpAdam/target/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
(
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*½'
value³'B°' B©'
û
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-0
layer-14
layer_with_weights-1
layer-15
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
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
R
	variables
trainable_variables
regularization_losses
	keras_api

axis
	gamma
beta
moving_mean
moving_variance
 	variables
!trainable_variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api

*iter

+beta_1

,beta_2
	-decay
.learning_ratemUmV$mW%mXvYvZ$v[%v\
*
0
1
2
3
$4
%5

0
1
$2
%3
 
­
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
 
­
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
 
[Y
VARIABLE_VALUEbatch_norm/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEbatch_norm/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEbatch_norm/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEbatch_norm/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
 
­
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
 	variables
!trainable_variables
"regularization_losses
YW
VARIABLE_VALUEtarget/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtarget/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
­
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
&	variables
'trainable_variables
(regularization_losses
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

0
1
v
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

C0
D1
E2
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
4
	Ftotal
	Gcount
H	variables
I	keras_api
D
	Jtotal
	Kcount
L
_fn_kwargs
M	variables
N	keras_api
p
Otrue_positives
Ptrue_negatives
Qfalse_positives
Rfalse_negatives
S	variables
T	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

F0
G1

H	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

M	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

O0
P1
Q2
R3

S	variables
~|
VARIABLE_VALUEAdam/batch_norm/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/batch_norm/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/target/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/target/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/batch_norm/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/batch_norm/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/target/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/target/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
v
serving_default_agePlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
u
serving_default_caPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
w
serving_default_cholPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
u
serving_default_cpPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
x
serving_default_exangPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_fbsPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
z
serving_default_oldpeakPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
z
serving_default_restecgPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
v
serving_default_sexPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
x
serving_default_slopePlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
w
serving_default_thalPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
z
serving_default_thalachPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
{
serving_default_trestbpsPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
È
StatefulPartitionedCallStatefulPartitionedCallserving_default_ageserving_default_caserving_default_cholserving_default_cpserving_default_exangserving_default_fbsserving_default_oldpeakserving_default_restecgserving_default_sexserving_default_slopeserving_default_thalserving_default_thalachserving_default_trestbpsbatch_norm/moving_variancebatch_norm/gammabatch_norm/moving_meanbatch_norm/betatarget/kerneltarget/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_4316
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$batch_norm/gamma/Read/ReadVariableOp#batch_norm/beta/Read/ReadVariableOp*batch_norm/moving_mean/Read/ReadVariableOp.batch_norm/moving_variance/Read/ReadVariableOp!target/kernel/Read/ReadVariableOptarget/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp+Adam/batch_norm/gamma/m/Read/ReadVariableOp*Adam/batch_norm/beta/m/Read/ReadVariableOp(Adam/target/kernel/m/Read/ReadVariableOp&Adam/target/bias/m/Read/ReadVariableOp+Adam/batch_norm/gamma/v/Read/ReadVariableOp*Adam/batch_norm/beta/v/Read/ReadVariableOp(Adam/target/kernel/v/Read/ReadVariableOp&Adam/target/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_4790
ÿ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_norm/gammabatch_norm/betabatch_norm/moving_meanbatch_norm/moving_variancetarget/kerneltarget/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1true_positivestrue_negativesfalse_positivesfalse_negativesAdam/batch_norm/gamma/mAdam/batch_norm/beta/mAdam/target/kernel/mAdam/target/bias/mAdam/batch_norm/gamma/vAdam/batch_norm/beta/vAdam/target/kernel/vAdam/target/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_4881÷ç
%
Ý
D__inference_batch_norm_layer_call_and_return_conditional_losses_3944

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
K
Ê
__inference__wrapped_model_3873
age
ca
chol
cp	
exang
fbs
oldpeak
restecg
sex	
slope
thal
thalach
trestbps@
2model_batch_norm_batchnorm_readvariableop_resource:D
6model_batch_norm_batchnorm_mul_readvariableop_resource:B
4model_batch_norm_batchnorm_readvariableop_1_resource:B
4model_batch_norm_batchnorm_readvariableop_2_resource:=
+model_target_matmul_readvariableop_resource::
,model_target_biasadd_readvariableop_resource:
identity¢)model/batch_norm/batchnorm/ReadVariableOp¢+model/batch_norm/batchnorm/ReadVariableOp_1¢+model/batch_norm/batchnorm/ReadVariableOp_2¢-model/batch_norm/batchnorm/mul/ReadVariableOp¢#model/target/BiasAdd/ReadVariableOp¢"model/target/MatMul/ReadVariableOpb
model/feature_layer/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB
model/feature_layer/GreaterGreaterage&model/feature_layer/Greater/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
model/feature_layer/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/feature_layer/EqualEqualsexmodel/feature_layer/x:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( 
model/feature_layer/LogicalAnd
LogicalAndmodel/feature_layer/Greater:z:0model/feature_layer/Equal:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/feature_layer/CastCast"model/feature_layer/LogicalAnd:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
model/feature_layer/x_1Const*
_output_shapes
: *
dtype0*
valueB Bfixed
model/feature_layer/Equal_1Equalthal model/feature_layer/x_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( 
model/feature_layer/Cast_1Castmodel/feature_layer/Equal_1:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
model/feature_layer/x_2Const*
_output_shapes
: *
dtype0*
valueB Bnormal
model/feature_layer/Equal_2Equalthal model/feature_layer/x_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( 
model/feature_layer/Cast_2Castmodel/feature_layer/Equal_2:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
model/feature_layer/x_3Const*
_output_shapes
: *
dtype0*
valueB B
reversible
model/feature_layer/Equal_3Equalthal model/feature_layer/x_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( 
model/feature_layer/Cast_3Castmodel/feature_layer/Equal_3:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
model/feature_layer/truedivRealDivtrestbpschol*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
model/feature_layer/mulMultrestbpsthalach*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8model/feature_layer/engineered_feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ù
3model/feature_layer/engineered_feature_layer/concatConcatV2model/feature_layer/Cast_1:y:0model/feature_layer/Cast_2:y:0model/feature_layer/Cast_3:y:0model/feature_layer/Cast:y:0model/feature_layer/truediv:z:0model/feature_layer/mul:z:0Amodel/feature_layer/engineered_feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
5model/feature_layer/numeric_feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ñ
0model/feature_layer/numeric_feature_layer/concatConcatV2trestbpscholthalacholdpeakslopecprestecgca>model/feature_layer/numeric_feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
4model/feature_layer/binary_feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Æ
/model/feature_layer/binary_feature_layer/concatConcatV2sexfbsexang=model/feature_layer/binary_feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
-model/feature_layer/feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ú
(model/feature_layer/feature_layer/concatConcatV2<model/feature_layer/engineered_feature_layer/concat:output:09model/feature_layer/numeric_feature_layer/concat:output:08model/feature_layer/binary_feature_layer/concat:output:06model/feature_layer/feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)model/batch_norm/batchnorm/ReadVariableOpReadVariableOp2model_batch_norm_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0e
 model/batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ª
model/batch_norm/batchnorm/addAddV21model/batch_norm/batchnorm/ReadVariableOp:value:0)model/batch_norm/batchnorm/add/y:output:0*
T0*
_output_shapes
:r
 model/batch_norm/batchnorm/RsqrtRsqrt"model/batch_norm/batchnorm/add:z:0*
T0*
_output_shapes
: 
-model/batch_norm/batchnorm/mul/ReadVariableOpReadVariableOp6model_batch_norm_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0§
model/batch_norm/batchnorm/mulMul$model/batch_norm/batchnorm/Rsqrt:y:05model/batch_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:°
 model/batch_norm/batchnorm/mul_1Mul1model/feature_layer/feature_layer/concat:output:0"model/batch_norm/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+model/batch_norm/batchnorm/ReadVariableOp_1ReadVariableOp4model_batch_norm_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0¥
 model/batch_norm/batchnorm/mul_2Mul3model/batch_norm/batchnorm/ReadVariableOp_1:value:0"model/batch_norm/batchnorm/mul:z:0*
T0*
_output_shapes
:
+model/batch_norm/batchnorm/ReadVariableOp_2ReadVariableOp4model_batch_norm_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0¥
model/batch_norm/batchnorm/subSub3model/batch_norm/batchnorm/ReadVariableOp_2:value:0$model/batch_norm/batchnorm/mul_2:z:0*
T0*
_output_shapes
:¥
 model/batch_norm/batchnorm/add_1AddV2$model/batch_norm/batchnorm/mul_1:z:0"model/batch_norm/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model/target/MatMul/ReadVariableOpReadVariableOp+model_target_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¡
model/target/MatMulMatMul$model/batch_norm/batchnorm/add_1:z:0*model/target/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#model/target/BiasAdd/ReadVariableOpReadVariableOp,model_target_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
model/target/BiasAddBiasAddmodel/target/MatMul:product:0+model/target/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
model/target/SigmoidSigmoidmodel/target/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitymodel/target/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
NoOpNoOp*^model/batch_norm/batchnorm/ReadVariableOp,^model/batch_norm/batchnorm/ReadVariableOp_1,^model/batch_norm/batchnorm/ReadVariableOp_2.^model/batch_norm/batchnorm/mul/ReadVariableOp$^model/target/BiasAdd/ReadVariableOp#^model/target/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 2V
)model/batch_norm/batchnorm/ReadVariableOp)model/batch_norm/batchnorm/ReadVariableOp2Z
+model/batch_norm/batchnorm/ReadVariableOp_1+model/batch_norm/batchnorm/ReadVariableOp_12Z
+model/batch_norm/batchnorm/ReadVariableOp_2+model/batch_norm/batchnorm/ReadVariableOp_22^
-model/batch_norm/batchnorm/mul/ReadVariableOp-model/batch_norm/batchnorm/mul/ReadVariableOp2J
#model/target/BiasAdd/ReadVariableOp#model/target/BiasAdd/ReadVariableOp2H
"model/target/MatMul/ReadVariableOp"model/target/MatMul/ReadVariableOp:L H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameage:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameca:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namechol:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namecp:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameexang:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namefbs:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	oldpeak:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	restecg:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namesex:N	J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameslope:M
I
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namethal:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	thalach:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
trestbps
Ñ
ó
$__inference_model_layer_call_fn_4071
age
ca
chol
cp	
exang
fbs
oldpeak
restecg
sex	
slope
thal
thalach
trestbps
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallagecacholcpexangfbsoldpeakrestecgsexslopethalthalachtrestbpsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_4056o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameage:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameca:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namechol:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namecp:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameexang:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namefbs:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	oldpeak:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	restecg:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namesex:N	J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameslope:M
I
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namethal:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	thalach:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
trestbps
%
Ý
D__inference_batch_norm_layer_call_and_return_conditional_losses_4654

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
£
D__inference_batch_norm_layer_call_and_return_conditional_losses_3897

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
Î
$__inference_model_layer_call_fn_4374

inputs_age
	inputs_ca
inputs_chol
	inputs_cp
inputs_exang

inputs_fbs
inputs_oldpeak
inputs_restecg

inputs_sex
inputs_slope
inputs_thal
inputs_thalach
inputs_trestbps
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCall
inputs_age	inputs_cainputs_chol	inputs_cpinputs_exang
inputs_fbsinputs_oldpeakinputs_restecg
inputs_sexinputs_slopeinputs_thalinputs_thalachinputs_trestbpsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_4173o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/age:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/ca:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinputs/chol:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/cp:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinputs/exang:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/fbs:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/oldpeak:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/restecg:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/sex:U	Q
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinputs/slope:T
P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinputs/thal:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/thalach:XT
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameinputs/trestbps

Ä
)__inference_batch_norm_layer_call_fn_4587

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_batch_norm_layer_call_and_return_conditional_losses_3897o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ$

G__inference_feature_layer_layer_call_and_return_conditional_losses_4027

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
identityN
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB`
GreaterGreaterinputsGreater/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?v
EqualEqualinputs_8
x:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( Y

LogicalAnd
LogicalAndGreater:z:0	Equal:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
CastCastLogicalAnd:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
x_1Const*
_output_shapes
: *
dtype0*
valueB Bfixed{
Equal_1Equal	inputs_10x_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( \
Cast_1CastEqual_1:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
x_2Const*
_output_shapes
: *
dtype0*
valueB Bnormal{
Equal_2Equal	inputs_10x_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( \
Cast_2CastEqual_2:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
x_3Const*
_output_shapes
: *
dtype0*
valueB B
reversible{
Equal_3Equal	inputs_10x_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( \
Cast_3CastEqual_3:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
truedivRealDiv	inputs_12inputs_2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
mulMul	inputs_12	inputs_11*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$engineered_feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ù
engineered_feature_layer/concatConcatV2
Cast_1:y:0
Cast_2:y:0
Cast_3:y:0Cast:y:0truediv:z:0mul:z:0-engineered_feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!numeric_feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :á
numeric_feature_layer/concatConcatV2	inputs_12inputs_2	inputs_11inputs_6inputs_9inputs_3inputs_7inputs_1*numeric_feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 binary_feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :«
binary_feature_layer/concatConcatV2inputs_8inputs_5inputs_4)binary_feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ö
feature_layer/concatConcatV2(engineered_feature_layer/concat:output:0%numeric_feature_layer/concat:output:0$binary_feature_layer/concat:output:0"feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
IdentityIdentityfeature_layer/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesú
÷:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:O	K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:O
K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
H
ý
?__inference_model_layer_call_and_return_conditional_losses_4438

inputs_age
	inputs_ca
inputs_chol
	inputs_cp
inputs_exang

inputs_fbs
inputs_oldpeak
inputs_restecg

inputs_sex
inputs_slope
inputs_thal
inputs_thalach
inputs_trestbps:
,batch_norm_batchnorm_readvariableop_resource:>
0batch_norm_batchnorm_mul_readvariableop_resource:<
.batch_norm_batchnorm_readvariableop_1_resource:<
.batch_norm_batchnorm_readvariableop_2_resource:7
%target_matmul_readvariableop_resource:4
&target_biasadd_readvariableop_resource:
identity¢#batch_norm/batchnorm/ReadVariableOp¢%batch_norm/batchnorm/ReadVariableOp_1¢%batch_norm/batchnorm/ReadVariableOp_2¢'batch_norm/batchnorm/mul/ReadVariableOp¢target/BiasAdd/ReadVariableOp¢target/MatMul/ReadVariableOp\
feature_layer/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB
feature_layer/GreaterGreater
inputs_age feature_layer/Greater/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
feature_layer/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
feature_layer/EqualEqual
inputs_sexfeature_layer/x:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( 
feature_layer/LogicalAnd
LogicalAndfeature_layer/Greater:z:0feature_layer/Equal:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
feature_layer/CastCastfeature_layer/LogicalAnd:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
feature_layer/x_1Const*
_output_shapes
: *
dtype0*
valueB Bfixed
feature_layer/Equal_1Equalinputs_thalfeature_layer/x_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( x
feature_layer/Cast_1Castfeature_layer/Equal_1:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
feature_layer/x_2Const*
_output_shapes
: *
dtype0*
valueB Bnormal
feature_layer/Equal_2Equalinputs_thalfeature_layer/x_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( x
feature_layer/Cast_2Castfeature_layer/Equal_2:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
feature_layer/x_3Const*
_output_shapes
: *
dtype0*
valueB B
reversible
feature_layer/Equal_3Equalinputs_thalfeature_layer/x_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( x
feature_layer/Cast_3Castfeature_layer/Equal_3:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
feature_layer/truedivRealDivinputs_trestbpsinputs_chol*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
feature_layer/mulMulinputs_trestbpsinputs_thalach*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
2feature_layer/engineered_feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :É
-feature_layer/engineered_feature_layer/concatConcatV2feature_layer/Cast_1:y:0feature_layer/Cast_2:y:0feature_layer/Cast_3:y:0feature_layer/Cast:y:0feature_layer/truediv:z:0feature_layer/mul:z:0;feature_layer/engineered_feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
/feature_layer/numeric_feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
*feature_layer/numeric_feature_layer/concatConcatV2inputs_trestbpsinputs_cholinputs_thalachinputs_oldpeakinputs_slope	inputs_cpinputs_restecg	inputs_ca8feature_layer/numeric_feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
.feature_layer/binary_feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ï
)feature_layer/binary_feature_layer/concatConcatV2
inputs_sex
inputs_fbsinputs_exang7feature_layer/binary_feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
'feature_layer/feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¼
"feature_layer/feature_layer/concatConcatV26feature_layer/engineered_feature_layer/concat:output:03feature_layer/numeric_feature_layer/concat:output:02feature_layer/binary_feature_layer/concat:output:00feature_layer/feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#batch_norm/batchnorm/ReadVariableOpReadVariableOp,batch_norm_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0_
batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batch_norm/batchnorm/addAddV2+batch_norm/batchnorm/ReadVariableOp:value:0#batch_norm/batchnorm/add/y:output:0*
T0*
_output_shapes
:f
batch_norm/batchnorm/RsqrtRsqrtbatch_norm/batchnorm/add:z:0*
T0*
_output_shapes
:
'batch_norm/batchnorm/mul/ReadVariableOpReadVariableOp0batch_norm_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batch_norm/batchnorm/mulMulbatch_norm/batchnorm/Rsqrt:y:0/batch_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
batch_norm/batchnorm/mul_1Mul+feature_layer/feature_layer/concat:output:0batch_norm/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%batch_norm/batchnorm/ReadVariableOp_1ReadVariableOp.batch_norm_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0
batch_norm/batchnorm/mul_2Mul-batch_norm/batchnorm/ReadVariableOp_1:value:0batch_norm/batchnorm/mul:z:0*
T0*
_output_shapes
:
%batch_norm/batchnorm/ReadVariableOp_2ReadVariableOp.batch_norm_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0
batch_norm/batchnorm/subSub-batch_norm/batchnorm/ReadVariableOp_2:value:0batch_norm/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batch_norm/batchnorm/add_1AddV2batch_norm/batchnorm/mul_1:z:0batch_norm/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
target/MatMul/ReadVariableOpReadVariableOp%target_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
target/MatMulMatMulbatch_norm/batchnorm/add_1:z:0$target/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
target/BiasAdd/ReadVariableOpReadVariableOp&target_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
target/BiasAddBiasAddtarget/MatMul:product:0%target/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
target/SigmoidSigmoidtarget/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentitytarget/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
NoOpNoOp$^batch_norm/batchnorm/ReadVariableOp&^batch_norm/batchnorm/ReadVariableOp_1&^batch_norm/batchnorm/ReadVariableOp_2(^batch_norm/batchnorm/mul/ReadVariableOp^target/BiasAdd/ReadVariableOp^target/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 2J
#batch_norm/batchnorm/ReadVariableOp#batch_norm/batchnorm/ReadVariableOp2N
%batch_norm/batchnorm/ReadVariableOp_1%batch_norm/batchnorm/ReadVariableOp_12N
%batch_norm/batchnorm/ReadVariableOp_2%batch_norm/batchnorm/ReadVariableOp_22R
'batch_norm/batchnorm/mul/ReadVariableOp'batch_norm/batchnorm/mul/ReadVariableOp2>
target/BiasAdd/ReadVariableOptarget/BiasAdd/ReadVariableOp2<
target/MatMul/ReadVariableOptarget/MatMul/ReadVariableOp:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/age:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/ca:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinputs/chol:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/cp:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinputs/exang:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/fbs:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/oldpeak:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/restecg:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/sex:U	Q
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinputs/slope:T
P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinputs/thal:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/thalach:XT
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameinputs/trestbps


ñ
@__inference_target_layer_call_and_return_conditional_losses_4049

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô%
¼
G__inference_feature_layer_layer_call_and_return_conditional_losses_4574

inputs_age
	inputs_ca
inputs_chol
	inputs_cp
inputs_exang

inputs_fbs
inputs_oldpeak
inputs_restecg

inputs_sex
inputs_slope
inputs_thal
inputs_thalach
inputs_trestbps
identityN
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HBd
GreaterGreater
inputs_ageGreater/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿF
xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
EqualEqual
inputs_sex
x:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( Y

LogicalAnd
LogicalAndGreater:z:0	Equal:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
CastCastLogicalAnd:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
x_1Const*
_output_shapes
: *
dtype0*
valueB Bfixed}
Equal_1Equalinputs_thalx_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( \
Cast_1CastEqual_1:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿJ
x_2Const*
_output_shapes
: *
dtype0*
valueB Bnormal}
Equal_2Equalinputs_thalx_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( \
Cast_2CastEqual_2:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿN
x_3Const*
_output_shapes
: *
dtype0*
valueB B
reversible}
Equal_3Equalinputs_thalx_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( \
Cast_3CastEqual_3:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
truedivRealDivinputs_trestbpsinputs_chol*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mulMulinputs_trestbpsinputs_thalach*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$engineered_feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ù
engineered_feature_layer/concatConcatV2
Cast_1:y:0
Cast_2:y:0
Cast_3:y:0Cast:y:0truediv:z:0mul:z:0-engineered_feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!numeric_feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
numeric_feature_layer/concatConcatV2inputs_trestbpsinputs_cholinputs_thalachinputs_oldpeakinputs_slope	inputs_cpinputs_restecg	inputs_ca*numeric_feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
 binary_feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :³
binary_feature_layer/concatConcatV2
inputs_sex
inputs_fbsinputs_exang)binary_feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ö
feature_layer/concatConcatV2(engineered_feature_layer/concat:output:0%numeric_feature_layer/concat:output:0$binary_feature_layer/concat:output:0"feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
IdentityIdentityfeature_layer/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesú
÷:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/age:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/ca:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinputs/chol:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/cp:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinputs/exang:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/fbs:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/oldpeak:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/restecg:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/sex:U	Q
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinputs/slope:T
P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinputs/thal:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/thalach:XT
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameinputs/trestbps


ñ
@__inference_target_layer_call_and_return_conditional_losses_4674

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
¡
,__inference_feature_layer_layer_call_fn_4533

inputs_age
	inputs_ca
inputs_chol
	inputs_cp
inputs_exang

inputs_fbs
inputs_oldpeak
inputs_restecg

inputs_sex
inputs_slope
inputs_thal
inputs_thalach
inputs_trestbps
identityç
PartitionedCallPartitionedCall
inputs_age	inputs_cainputs_chol	inputs_cpinputs_exang
inputs_fbsinputs_oldpeakinputs_restecg
inputs_sexinputs_slopeinputs_thalinputs_thalachinputs_trestbps*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_feature_layer_layer_call_and_return_conditional_losses_4027`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesú
÷:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/age:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/ca:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinputs/chol:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/cp:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinputs/exang:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/fbs:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/oldpeak:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/restecg:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/sex:U	Q
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinputs/slope:T
P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinputs/thal:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/thalach:XT
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameinputs/trestbps
àk
Ó
 __inference__traced_restore_4881
file_prefix/
!assignvariableop_batch_norm_gamma:0
"assignvariableop_1_batch_norm_beta:7
)assignvariableop_2_batch_norm_moving_mean:;
-assignvariableop_3_batch_norm_moving_variance:2
 assignvariableop_4_target_kernel:,
assignvariableop_5_target_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: 1
"assignvariableop_15_true_positives:	È1
"assignvariableop_16_true_negatives:	È2
#assignvariableop_17_false_positives:	È2
#assignvariableop_18_false_negatives:	È9
+assignvariableop_19_adam_batch_norm_gamma_m:8
*assignvariableop_20_adam_batch_norm_beta_m::
(assignvariableop_21_adam_target_kernel_m:4
&assignvariableop_22_adam_target_bias_m:9
+assignvariableop_23_adam_batch_norm_gamma_v:8
*assignvariableop_24_adam_batch_norm_beta_v::
(assignvariableop_25_adam_target_kernel_v:4
&assignvariableop_26_adam_target_bias_v:
identity_28¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9·
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ý
valueÓBÐB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¨
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B «
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_batch_norm_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_batch_norm_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp)assignvariableop_2_batch_norm_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_norm_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp assignvariableop_4_target_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_target_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp"assignvariableop_15_true_positivesIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp"assignvariableop_16_true_negativesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp#assignvariableop_17_false_positivesIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp#assignvariableop_18_false_negativesIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_batch_norm_gamma_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_batch_norm_beta_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_target_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_target_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_batch_norm_gamma_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_batch_norm_beta_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_target_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_target_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¡
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
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
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
º

%__inference_target_layer_call_fn_4663

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_target_layer_call_and_return_conditional_losses_4049o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
£
D__inference_batch_norm_layer_call_and_return_conditional_losses_4620

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
Î
$__inference_model_layer_call_fn_4345

inputs_age
	inputs_ca
inputs_chol
	inputs_cp
inputs_exang

inputs_fbs
inputs_oldpeak
inputs_restecg

inputs_sex
inputs_slope
inputs_thal
inputs_thalach
inputs_trestbps
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCall
inputs_age	inputs_cainputs_chol	inputs_cpinputs_exang
inputs_fbsinputs_oldpeakinputs_restecg
inputs_sexinputs_slopeinputs_thalinputs_thalachinputs_trestbpsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_4056o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/age:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/ca:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinputs/chol:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/cp:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinputs/exang:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/fbs:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/oldpeak:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/restecg:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/sex:U	Q
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinputs/slope:T
P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinputs/thal:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/thalach:XT
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameinputs/trestbps
©

?__inference_model_layer_call_and_return_conditional_losses_4173

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
batch_norm_4158:
batch_norm_4160:
batch_norm_4162:
batch_norm_4164:
target_4167:
target_4169:
identity¢"batch_norm/StatefulPartitionedCall¢target/StatefulPartitionedCallÇ
feature_layer/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_feature_layer_layer_call_and_return_conditional_losses_4027¶
"batch_norm/StatefulPartitionedCallStatefulPartitionedCall&feature_layer/PartitionedCall:output:0batch_norm_4158batch_norm_4160batch_norm_4162batch_norm_4164*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_batch_norm_layer_call_and_return_conditional_losses_3944
target/StatefulPartitionedCallStatefulPartitionedCall+batch_norm/StatefulPartitionedCall:output:0target_4167target_4169*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_target_layer_call_and_return_conditional_losses_4049v
IdentityIdentity'target/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp#^batch_norm/StatefulPartitionedCall^target/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"batch_norm/StatefulPartitionedCall"batch_norm/StatefulPartitionedCall2@
target/StatefulPartitionedCalltarget/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:O	K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:O
K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
^
Í
?__inference_model_layer_call_and_return_conditional_losses_4516

inputs_age
	inputs_ca
inputs_chol
	inputs_cp
inputs_exang

inputs_fbs
inputs_oldpeak
inputs_restecg

inputs_sex
inputs_slope
inputs_thal
inputs_thalach
inputs_trestbps@
2batch_norm_assignmovingavg_readvariableop_resource:B
4batch_norm_assignmovingavg_1_readvariableop_resource:>
0batch_norm_batchnorm_mul_readvariableop_resource::
,batch_norm_batchnorm_readvariableop_resource:7
%target_matmul_readvariableop_resource:4
&target_biasadd_readvariableop_resource:
identity¢batch_norm/AssignMovingAvg¢)batch_norm/AssignMovingAvg/ReadVariableOp¢batch_norm/AssignMovingAvg_1¢+batch_norm/AssignMovingAvg_1/ReadVariableOp¢#batch_norm/batchnorm/ReadVariableOp¢'batch_norm/batchnorm/mul/ReadVariableOp¢target/BiasAdd/ReadVariableOp¢target/MatMul/ReadVariableOp\
feature_layer/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *  HB
feature_layer/GreaterGreater
inputs_age feature_layer/Greater/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
feature_layer/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
feature_layer/EqualEqual
inputs_sexfeature_layer/x:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( 
feature_layer/LogicalAnd
LogicalAndfeature_layer/Greater:z:0feature_layer/Equal:z:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
feature_layer/CastCastfeature_layer/LogicalAnd:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
feature_layer/x_1Const*
_output_shapes
: *
dtype0*
valueB Bfixed
feature_layer/Equal_1Equalinputs_thalfeature_layer/x_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( x
feature_layer/Cast_1Castfeature_layer/Equal_1:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
feature_layer/x_2Const*
_output_shapes
: *
dtype0*
valueB Bnormal
feature_layer/Equal_2Equalinputs_thalfeature_layer/x_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( x
feature_layer/Cast_2Castfeature_layer/Equal_2:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
feature_layer/x_3Const*
_output_shapes
: *
dtype0*
valueB B
reversible
feature_layer/Equal_3Equalinputs_thalfeature_layer/x_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
incompatible_shape_error( x
feature_layer/Cast_3Castfeature_layer/Equal_3:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
feature_layer/truedivRealDivinputs_trestbpsinputs_chol*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
feature_layer/mulMulinputs_trestbpsinputs_thalach*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
2feature_layer/engineered_feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :É
-feature_layer/engineered_feature_layer/concatConcatV2feature_layer/Cast_1:y:0feature_layer/Cast_2:y:0feature_layer/Cast_3:y:0feature_layer/Cast:y:0feature_layer/truediv:z:0feature_layer/mul:z:0;feature_layer/engineered_feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
/feature_layer/numeric_feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
*feature_layer/numeric_feature_layer/concatConcatV2inputs_trestbpsinputs_cholinputs_thalachinputs_oldpeakinputs_slope	inputs_cpinputs_restecg	inputs_ca8feature_layer/numeric_feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
.feature_layer/binary_feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ï
)feature_layer/binary_feature_layer/concatConcatV2
inputs_sex
inputs_fbsinputs_exang7feature_layer/binary_feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
'feature_layer/feature_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¼
"feature_layer/feature_layer/concatConcatV26feature_layer/engineered_feature_layer/concat:output:03feature_layer/numeric_feature_layer/concat:output:02feature_layer/binary_feature_layer/concat:output:00feature_layer/feature_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
)batch_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: º
batch_norm/moments/meanMean+feature_layer/feature_layer/concat:output:02batch_norm/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(z
batch_norm/moments/StopGradientStopGradient batch_norm/moments/mean:output:0*
T0*
_output_shapes

:Â
$batch_norm/moments/SquaredDifferenceSquaredDifference+feature_layer/feature_layer/concat:output:0(batch_norm/moments/StopGradient:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
-batch_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¿
batch_norm/moments/varianceMean(batch_norm/moments/SquaredDifference:z:06batch_norm/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(
batch_norm/moments/SqueezeSqueeze batch_norm/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 
batch_norm/moments/Squeeze_1Squeeze$batch_norm/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 e
 batch_norm/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
)batch_norm/AssignMovingAvg/ReadVariableOpReadVariableOp2batch_norm_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0¢
batch_norm/AssignMovingAvg/subSub1batch_norm/AssignMovingAvg/ReadVariableOp:value:0#batch_norm/moments/Squeeze:output:0*
T0*
_output_shapes
:
batch_norm/AssignMovingAvg/mulMul"batch_norm/AssignMovingAvg/sub:z:0)batch_norm/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:Ø
batch_norm/AssignMovingAvgAssignSubVariableOp2batch_norm_assignmovingavg_readvariableop_resource"batch_norm/AssignMovingAvg/mul:z:0*^batch_norm/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0g
"batch_norm/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
+batch_norm/AssignMovingAvg_1/ReadVariableOpReadVariableOp4batch_norm_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0¨
 batch_norm/AssignMovingAvg_1/subSub3batch_norm/AssignMovingAvg_1/ReadVariableOp:value:0%batch_norm/moments/Squeeze_1:output:0*
T0*
_output_shapes
:
 batch_norm/AssignMovingAvg_1/mulMul$batch_norm/AssignMovingAvg_1/sub:z:0+batch_norm/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:à
batch_norm/AssignMovingAvg_1AssignSubVariableOp4batch_norm_assignmovingavg_1_readvariableop_resource$batch_norm/AssignMovingAvg_1/mul:z:0,^batch_norm/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0_
batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batch_norm/batchnorm/addAddV2%batch_norm/moments/Squeeze_1:output:0#batch_norm/batchnorm/add/y:output:0*
T0*
_output_shapes
:f
batch_norm/batchnorm/RsqrtRsqrtbatch_norm/batchnorm/add:z:0*
T0*
_output_shapes
:
'batch_norm/batchnorm/mul/ReadVariableOpReadVariableOp0batch_norm_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0
batch_norm/batchnorm/mulMulbatch_norm/batchnorm/Rsqrt:y:0/batch_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:
batch_norm/batchnorm/mul_1Mul+feature_layer/feature_layer/concat:output:0batch_norm/batchnorm/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
batch_norm/batchnorm/mul_2Mul#batch_norm/moments/Squeeze:output:0batch_norm/batchnorm/mul:z:0*
T0*
_output_shapes
:
#batch_norm/batchnorm/ReadVariableOpReadVariableOp,batch_norm_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0
batch_norm/batchnorm/subSub+batch_norm/batchnorm/ReadVariableOp:value:0batch_norm/batchnorm/mul_2:z:0*
T0*
_output_shapes
:
batch_norm/batchnorm/add_1AddV2batch_norm/batchnorm/mul_1:z:0batch_norm/batchnorm/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
target/MatMul/ReadVariableOpReadVariableOp%target_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
target/MatMulMatMulbatch_norm/batchnorm/add_1:z:0$target/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
target/BiasAdd/ReadVariableOpReadVariableOp&target_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
target/BiasAddBiasAddtarget/MatMul:product:0%target/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
target/SigmoidSigmoidtarget/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentitytarget/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿë
NoOpNoOp^batch_norm/AssignMovingAvg*^batch_norm/AssignMovingAvg/ReadVariableOp^batch_norm/AssignMovingAvg_1,^batch_norm/AssignMovingAvg_1/ReadVariableOp$^batch_norm/batchnorm/ReadVariableOp(^batch_norm/batchnorm/mul/ReadVariableOp^target/BiasAdd/ReadVariableOp^target/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 28
batch_norm/AssignMovingAvgbatch_norm/AssignMovingAvg2V
)batch_norm/AssignMovingAvg/ReadVariableOp)batch_norm/AssignMovingAvg/ReadVariableOp2<
batch_norm/AssignMovingAvg_1batch_norm/AssignMovingAvg_12Z
+batch_norm/AssignMovingAvg_1/ReadVariableOp+batch_norm/AssignMovingAvg_1/ReadVariableOp2J
#batch_norm/batchnorm/ReadVariableOp#batch_norm/batchnorm/ReadVariableOp2R
'batch_norm/batchnorm/mul/ReadVariableOp'batch_norm/batchnorm/mul/ReadVariableOp2>
target/BiasAdd/ReadVariableOptarget/BiasAdd/ReadVariableOp2<
target/MatMul/ReadVariableOptarget/MatMul/ReadVariableOp:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/age:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/ca:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinputs/chol:RN
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#
_user_specified_name	inputs/cp:UQ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinputs/exang:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/fbs:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/oldpeak:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/restecg:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
inputs/sex:U	Q
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameinputs/slope:T
P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameinputs/thal:WS
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_nameinputs/thalach:XT
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameinputs/trestbps
¿
Ø
?__inference_model_layer_call_and_return_conditional_losses_4248
age
ca
chol
cp	
exang
fbs
oldpeak
restecg
sex	
slope
thal
thalach
trestbps
batch_norm_4233:
batch_norm_4235:
batch_norm_4237:
batch_norm_4239:
target_4242:
target_4244:
identity¢"batch_norm/StatefulPartitionedCall¢target/StatefulPartitionedCall
feature_layer/PartitionedCallPartitionedCallagecacholcpexangfbsoldpeakrestecgsexslopethalthalachtrestbps*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_feature_layer_layer_call_and_return_conditional_losses_4027¸
"batch_norm/StatefulPartitionedCallStatefulPartitionedCall&feature_layer/PartitionedCall:output:0batch_norm_4233batch_norm_4235batch_norm_4237batch_norm_4239*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_batch_norm_layer_call_and_return_conditional_losses_3897
target/StatefulPartitionedCallStatefulPartitionedCall+batch_norm/StatefulPartitionedCall:output:0target_4242target_4244*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_target_layer_call_and_return_conditional_losses_4049v
IdentityIdentity'target/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp#^batch_norm/StatefulPartitionedCall^target/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"batch_norm/StatefulPartitionedCall"batch_norm/StatefulPartitionedCall2@
target/StatefulPartitionedCalltarget/StatefulPartitionedCall:L H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameage:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameca:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namechol:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namecp:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameexang:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namefbs:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	oldpeak:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	restecg:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namesex:N	J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameslope:M
I
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namethal:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	thalach:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
trestbps

Ä
)__inference_batch_norm_layer_call_fn_4600

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_batch_norm_layer_call_and_return_conditional_losses_3944o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
Ø
?__inference_model_layer_call_and_return_conditional_losses_4279
age
ca
chol
cp	
exang
fbs
oldpeak
restecg
sex	
slope
thal
thalach
trestbps
batch_norm_4264:
batch_norm_4266:
batch_norm_4268:
batch_norm_4270:
target_4273:
target_4275:
identity¢"batch_norm/StatefulPartitionedCall¢target/StatefulPartitionedCall
feature_layer/PartitionedCallPartitionedCallagecacholcpexangfbsoldpeakrestecgsexslopethalthalachtrestbps*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_feature_layer_layer_call_and_return_conditional_losses_4027¶
"batch_norm/StatefulPartitionedCallStatefulPartitionedCall&feature_layer/PartitionedCall:output:0batch_norm_4264batch_norm_4266batch_norm_4268batch_norm_4270*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_batch_norm_layer_call_and_return_conditional_losses_3944
target/StatefulPartitionedCallStatefulPartitionedCall+batch_norm/StatefulPartitionedCall:output:0target_4273target_4275*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_target_layer_call_and_return_conditional_losses_4049v
IdentityIdentity'target/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp#^batch_norm/StatefulPartitionedCall^target/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"batch_norm/StatefulPartitionedCall"batch_norm/StatefulPartitionedCall2@
target/StatefulPartitionedCalltarget/StatefulPartitionedCall:L H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameage:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameca:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namechol:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namecp:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameexang:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namefbs:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	oldpeak:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	restecg:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namesex:N	J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameslope:M
I
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namethal:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	thalach:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
trestbps
¯
ñ
"__inference_signature_wrapper_4316
age
ca
chol
cp	
exang
fbs
oldpeak
restecg
sex	
slope
thal
thalach
trestbps
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallagecacholcpexangfbsoldpeakrestecgsexslopethalthalachtrestbpsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_3873o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameage:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameca:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namechol:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namecp:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameexang:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namefbs:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	oldpeak:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	restecg:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namesex:N	J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameslope:M
I
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namethal:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	thalach:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
trestbps
Ý:
ÿ

__inference__traced_save_4790
file_prefix/
+savev2_batch_norm_gamma_read_readvariableop.
*savev2_batch_norm_beta_read_readvariableop5
1savev2_batch_norm_moving_mean_read_readvariableop9
5savev2_batch_norm_moving_variance_read_readvariableop,
(savev2_target_kernel_read_readvariableop*
&savev2_target_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop6
2savev2_adam_batch_norm_gamma_m_read_readvariableop5
1savev2_adam_batch_norm_beta_m_read_readvariableop3
/savev2_adam_target_kernel_m_read_readvariableop1
-savev2_adam_target_bias_m_read_readvariableop6
2savev2_adam_batch_norm_gamma_v_read_readvariableop5
1savev2_adam_batch_norm_beta_v_read_readvariableop3
/savev2_adam_target_kernel_v_read_readvariableop1
-savev2_adam_target_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ´
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ý
valueÓBÐB5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¥
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B ô

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_batch_norm_gamma_read_readvariableop*savev2_batch_norm_beta_read_readvariableop1savev2_batch_norm_moving_mean_read_readvariableop5savev2_batch_norm_moving_variance_read_readvariableop(savev2_target_kernel_read_readvariableop&savev2_target_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop2savev2_adam_batch_norm_gamma_m_read_readvariableop1savev2_adam_batch_norm_beta_m_read_readvariableop/savev2_adam_target_kernel_m_read_readvariableop-savev2_adam_target_bias_m_read_readvariableop2savev2_adam_batch_norm_gamma_v_read_readvariableop1savev2_adam_batch_norm_beta_v_read_readvariableop/savev2_adam_target_kernel_v_read_readvariableop-savev2_adam_target_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*§
_input_shapes
: ::::::: : : : : : : : : :È:È:È:È::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:È:!

_output_shapes	
:È:!

_output_shapes	
:È:!

_output_shapes	
:È: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
Ï
ó
$__inference_model_layer_call_fn_4217
age
ca
chol
cp	
exang
fbs
oldpeak
restecg
sex	
slope
thal
thalach
trestbps
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallagecacholcpexangfbsoldpeakrestecgsexslopethalthalachtrestbpsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_4173o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameage:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameca:MI
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namechol:KG
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namecp:NJ
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameexang:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namefbs:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	oldpeak:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	restecg:LH
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namesex:N	J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_nameslope:M
I
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namethal:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	thalach:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
trestbps
«

?__inference_model_layer_call_and_return_conditional_losses_4056

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
batch_norm_4029:
batch_norm_4031:
batch_norm_4033:
batch_norm_4035:
target_4050:
target_4052:
identity¢"batch_norm/StatefulPartitionedCall¢target/StatefulPartitionedCallÇ
feature_layer/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_feature_layer_layer_call_and_return_conditional_losses_4027¸
"batch_norm/StatefulPartitionedCallStatefulPartitionedCall&feature_layer/PartitionedCall:output:0batch_norm_4029batch_norm_4031batch_norm_4033batch_norm_4035*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_batch_norm_layer_call_and_return_conditional_losses_3897
target/StatefulPartitionedCallStatefulPartitionedCall+batch_norm/StatefulPartitionedCall:output:0target_4050target_4052*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_target_layer_call_and_return_conditional_losses_4049v
IdentityIdentity'target/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp#^batch_norm/StatefulPartitionedCall^target/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"batch_norm/StatefulPartitionedCall"batch_norm/StatefulPartitionedCall2@
target/StatefulPartitionedCalltarget/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:O	K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:O
K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ç
serving_default³
3
age,
serving_default_age:0ÿÿÿÿÿÿÿÿÿ
1
ca+
serving_default_ca:0ÿÿÿÿÿÿÿÿÿ
5
chol-
serving_default_chol:0ÿÿÿÿÿÿÿÿÿ
1
cp+
serving_default_cp:0ÿÿÿÿÿÿÿÿÿ
7
exang.
serving_default_exang:0ÿÿÿÿÿÿÿÿÿ
3
fbs,
serving_default_fbs:0ÿÿÿÿÿÿÿÿÿ
;
oldpeak0
serving_default_oldpeak:0ÿÿÿÿÿÿÿÿÿ
;
restecg0
serving_default_restecg:0ÿÿÿÿÿÿÿÿÿ
3
sex,
serving_default_sex:0ÿÿÿÿÿÿÿÿÿ
7
slope.
serving_default_slope:0ÿÿÿÿÿÿÿÿÿ
5
thal-
serving_default_thal:0ÿÿÿÿÿÿÿÿÿ
;
thalach0
serving_default_thalach:0ÿÿÿÿÿÿÿÿÿ
=
trestbps1
serving_default_trestbps:0ÿÿÿÿÿÿÿÿÿ:
target0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:
í
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-0
layer-14
layer_with_weights-1
layer-15
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
]__call__
*^&call_and_return_all_conditional_losses
__default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
¥
	variables
trainable_variables
regularization_losses
	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
axis
	gamma
beta
moving_mean
moving_variance
 	variables
!trainable_variables
"regularization_losses
#	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
»

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_layer

*iter

+beta_1

,beta_2
	-decay
.learning_ratemUmV$mW%mXvYvZ$v[%v\"
	optimizer
J
0
1
2
3
$4
%5"
trackable_list_wrapper
<
0
1
$2
%3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
]__call__
__default_save_signature
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
,
fserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
4non_trainable_variables

5layers
6metrics
7layer_regularization_losses
8layer_metrics
	variables
trainable_variables
regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:2batch_norm/gamma
:2batch_norm/beta
&:$ (2batch_norm/moving_mean
*:( (2batch_norm/moving_variance
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
9non_trainable_variables

:layers
;metrics
<layer_regularization_losses
=layer_metrics
 	variables
!trainable_variables
"regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
:2target/kernel
:2target/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
&	variables
'trainable_variables
(regularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
0
1"
trackable_list_wrapper

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
15"
trackable_list_wrapper
5
C0
D1
E2"
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
.
0
1"
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
N
	Ftotal
	Gcount
H	variables
I	keras_api"
_tf_keras_metric
^
	Jtotal
	Kcount
L
_fn_kwargs
M	variables
N	keras_api"
_tf_keras_metric

Otrue_positives
Ptrue_negatives
Qfalse_positives
Rfalse_negatives
S	variables
T	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
F0
G1"
trackable_list_wrapper
-
H	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
J0
K1"
trackable_list_wrapper
-
M	variables"
_generic_user_object
:È (2true_positives
:È (2true_negatives
 :È (2false_positives
 :È (2false_negatives
<
O0
P1
Q2
R3"
trackable_list_wrapper
-
S	variables"
_generic_user_object
#:!2Adam/batch_norm/gamma/m
": 2Adam/batch_norm/beta/m
$:"2Adam/target/kernel/m
:2Adam/target/bias/m
#:!2Adam/batch_norm/gamma/v
": 2Adam/batch_norm/beta/v
$:"2Adam/target/kernel/v
:2Adam/target/bias/v
Þ2Û
$__inference_model_layer_call_fn_4071
$__inference_model_layer_call_fn_4345
$__inference_model_layer_call_fn_4374
$__inference_model_layer_call_fn_4217À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
?__inference_model_layer_call_and_return_conditional_losses_4438
?__inference_model_layer_call_and_return_conditional_losses_4516
?__inference_model_layer_call_and_return_conditional_losses_4248
?__inference_model_layer_call_and_return_conditional_losses_4279À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
__inference__wrapped_model_3873agecacholcpexangfbsoldpeakrestecgsexslopethalthalachtrestbps"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_feature_layer_layer_call_fn_4533¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_feature_layer_layer_call_and_return_conditional_losses_4574¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
)__inference_batch_norm_layer_call_fn_4587
)__inference_batch_norm_layer_call_fn_4600´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_batch_norm_layer_call_and_return_conditional_losses_4620
D__inference_batch_norm_layer_call_and_return_conditional_losses_4654´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ï2Ì
%__inference_target_layer_call_fn_4663¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_target_layer_call_and_return_conditional_losses_4674¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
"__inference_signature_wrapper_4316agecacholcpexangfbsoldpeakrestecgsexslopethalthalachtrestbps"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
__inference__wrapped_model_3873ë$%¯¢«
£¢
ª
$
age
ageÿÿÿÿÿÿÿÿÿ
"
ca
caÿÿÿÿÿÿÿÿÿ
&
chol
cholÿÿÿÿÿÿÿÿÿ
"
cp
cpÿÿÿÿÿÿÿÿÿ
(
exang
exangÿÿÿÿÿÿÿÿÿ
$
fbs
fbsÿÿÿÿÿÿÿÿÿ
,
oldpeak!
oldpeakÿÿÿÿÿÿÿÿÿ
,
restecg!
restecgÿÿÿÿÿÿÿÿÿ
$
sex
sexÿÿÿÿÿÿÿÿÿ
(
slope
slopeÿÿÿÿÿÿÿÿÿ
&
thal
thalÿÿÿÿÿÿÿÿÿ
,
thalach!
thalachÿÿÿÿÿÿÿÿÿ
.
trestbps"
trestbpsÿÿÿÿÿÿÿÿÿ
ª "/ª,
*
target 
targetÿÿÿÿÿÿÿÿÿª
D__inference_batch_norm_layer_call_and_return_conditional_losses_4620b3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ª
D__inference_batch_norm_layer_call_and_return_conditional_losses_4654b3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_batch_norm_layer_call_fn_4587U3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_batch_norm_layer_call_fn_4600U3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
G__inference_feature_layer_layer_call_and_return_conditional_losses_4574´¢
þ¢ú
÷ªó
+
age$!

inputs/ageÿÿÿÿÿÿÿÿÿ
)
ca# 
	inputs/caÿÿÿÿÿÿÿÿÿ
-
chol%"
inputs/cholÿÿÿÿÿÿÿÿÿ
)
cp# 
	inputs/cpÿÿÿÿÿÿÿÿÿ
/
exang&#
inputs/exangÿÿÿÿÿÿÿÿÿ
+
fbs$!

inputs/fbsÿÿÿÿÿÿÿÿÿ
3
oldpeak(%
inputs/oldpeakÿÿÿÿÿÿÿÿÿ
3
restecg(%
inputs/restecgÿÿÿÿÿÿÿÿÿ
+
sex$!

inputs/sexÿÿÿÿÿÿÿÿÿ
/
slope&#
inputs/slopeÿÿÿÿÿÿÿÿÿ
-
thal%"
inputs/thalÿÿÿÿÿÿÿÿÿ
3
thalach(%
inputs/thalachÿÿÿÿÿÿÿÿÿ
5
trestbps)&
inputs/trestbpsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ø
,__inference_feature_layer_layer_call_fn_4533§¢
þ¢ú
÷ªó
+
age$!

inputs/ageÿÿÿÿÿÿÿÿÿ
)
ca# 
	inputs/caÿÿÿÿÿÿÿÿÿ
-
chol%"
inputs/cholÿÿÿÿÿÿÿÿÿ
)
cp# 
	inputs/cpÿÿÿÿÿÿÿÿÿ
/
exang&#
inputs/exangÿÿÿÿÿÿÿÿÿ
+
fbs$!

inputs/fbsÿÿÿÿÿÿÿÿÿ
3
oldpeak(%
inputs/oldpeakÿÿÿÿÿÿÿÿÿ
3
restecg(%
inputs/restecgÿÿÿÿÿÿÿÿÿ
+
sex$!

inputs/sexÿÿÿÿÿÿÿÿÿ
/
slope&#
inputs/slopeÿÿÿÿÿÿÿÿÿ
-
thal%"
inputs/thalÿÿÿÿÿÿÿÿÿ
3
thalach(%
inputs/thalachÿÿÿÿÿÿÿÿÿ
5
trestbps)&
inputs/trestbpsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ­
?__inference_model_layer_call_and_return_conditional_losses_4248é$%·¢³
«¢§
ª
$
age
ageÿÿÿÿÿÿÿÿÿ
"
ca
caÿÿÿÿÿÿÿÿÿ
&
chol
cholÿÿÿÿÿÿÿÿÿ
"
cp
cpÿÿÿÿÿÿÿÿÿ
(
exang
exangÿÿÿÿÿÿÿÿÿ
$
fbs
fbsÿÿÿÿÿÿÿÿÿ
,
oldpeak!
oldpeakÿÿÿÿÿÿÿÿÿ
,
restecg!
restecgÿÿÿÿÿÿÿÿÿ
$
sex
sexÿÿÿÿÿÿÿÿÿ
(
slope
slopeÿÿÿÿÿÿÿÿÿ
&
thal
thalÿÿÿÿÿÿÿÿÿ
,
thalach!
thalachÿÿÿÿÿÿÿÿÿ
.
trestbps"
trestbpsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ­
?__inference_model_layer_call_and_return_conditional_losses_4279é$%·¢³
«¢§
ª
$
age
ageÿÿÿÿÿÿÿÿÿ
"
ca
caÿÿÿÿÿÿÿÿÿ
&
chol
cholÿÿÿÿÿÿÿÿÿ
"
cp
cpÿÿÿÿÿÿÿÿÿ
(
exang
exangÿÿÿÿÿÿÿÿÿ
$
fbs
fbsÿÿÿÿÿÿÿÿÿ
,
oldpeak!
oldpeakÿÿÿÿÿÿÿÿÿ
,
restecg!
restecgÿÿÿÿÿÿÿÿÿ
$
sex
sexÿÿÿÿÿÿÿÿÿ
(
slope
slopeÿÿÿÿÿÿÿÿÿ
&
thal
thalÿÿÿÿÿÿÿÿÿ
,
thalach!
thalachÿÿÿÿÿÿÿÿÿ
.
trestbps"
trestbpsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
?__inference_model_layer_call_and_return_conditional_losses_4438Ä$%¢
¢
÷ªó
+
age$!

inputs/ageÿÿÿÿÿÿÿÿÿ
)
ca# 
	inputs/caÿÿÿÿÿÿÿÿÿ
-
chol%"
inputs/cholÿÿÿÿÿÿÿÿÿ
)
cp# 
	inputs/cpÿÿÿÿÿÿÿÿÿ
/
exang&#
inputs/exangÿÿÿÿÿÿÿÿÿ
+
fbs$!

inputs/fbsÿÿÿÿÿÿÿÿÿ
3
oldpeak(%
inputs/oldpeakÿÿÿÿÿÿÿÿÿ
3
restecg(%
inputs/restecgÿÿÿÿÿÿÿÿÿ
+
sex$!

inputs/sexÿÿÿÿÿÿÿÿÿ
/
slope&#
inputs/slopeÿÿÿÿÿÿÿÿÿ
-
thal%"
inputs/thalÿÿÿÿÿÿÿÿÿ
3
thalach(%
inputs/thalachÿÿÿÿÿÿÿÿÿ
5
trestbps)&
inputs/trestbpsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
?__inference_model_layer_call_and_return_conditional_losses_4516Ä$%¢
¢
÷ªó
+
age$!

inputs/ageÿÿÿÿÿÿÿÿÿ
)
ca# 
	inputs/caÿÿÿÿÿÿÿÿÿ
-
chol%"
inputs/cholÿÿÿÿÿÿÿÿÿ
)
cp# 
	inputs/cpÿÿÿÿÿÿÿÿÿ
/
exang&#
inputs/exangÿÿÿÿÿÿÿÿÿ
+
fbs$!

inputs/fbsÿÿÿÿÿÿÿÿÿ
3
oldpeak(%
inputs/oldpeakÿÿÿÿÿÿÿÿÿ
3
restecg(%
inputs/restecgÿÿÿÿÿÿÿÿÿ
+
sex$!

inputs/sexÿÿÿÿÿÿÿÿÿ
/
slope&#
inputs/slopeÿÿÿÿÿÿÿÿÿ
-
thal%"
inputs/thalÿÿÿÿÿÿÿÿÿ
3
thalach(%
inputs/thalachÿÿÿÿÿÿÿÿÿ
5
trestbps)&
inputs/trestbpsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
$__inference_model_layer_call_fn_4071Ü$%·¢³
«¢§
ª
$
age
ageÿÿÿÿÿÿÿÿÿ
"
ca
caÿÿÿÿÿÿÿÿÿ
&
chol
cholÿÿÿÿÿÿÿÿÿ
"
cp
cpÿÿÿÿÿÿÿÿÿ
(
exang
exangÿÿÿÿÿÿÿÿÿ
$
fbs
fbsÿÿÿÿÿÿÿÿÿ
,
oldpeak!
oldpeakÿÿÿÿÿÿÿÿÿ
,
restecg!
restecgÿÿÿÿÿÿÿÿÿ
$
sex
sexÿÿÿÿÿÿÿÿÿ
(
slope
slopeÿÿÿÿÿÿÿÿÿ
&
thal
thalÿÿÿÿÿÿÿÿÿ
,
thalach!
thalachÿÿÿÿÿÿÿÿÿ
.
trestbps"
trestbpsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
$__inference_model_layer_call_fn_4217Ü$%·¢³
«¢§
ª
$
age
ageÿÿÿÿÿÿÿÿÿ
"
ca
caÿÿÿÿÿÿÿÿÿ
&
chol
cholÿÿÿÿÿÿÿÿÿ
"
cp
cpÿÿÿÿÿÿÿÿÿ
(
exang
exangÿÿÿÿÿÿÿÿÿ
$
fbs
fbsÿÿÿÿÿÿÿÿÿ
,
oldpeak!
oldpeakÿÿÿÿÿÿÿÿÿ
,
restecg!
restecgÿÿÿÿÿÿÿÿÿ
$
sex
sexÿÿÿÿÿÿÿÿÿ
(
slope
slopeÿÿÿÿÿÿÿÿÿ
&
thal
thalÿÿÿÿÿÿÿÿÿ
,
thalach!
thalachÿÿÿÿÿÿÿÿÿ
.
trestbps"
trestbpsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿà
$__inference_model_layer_call_fn_4345·$%¢
¢
÷ªó
+
age$!

inputs/ageÿÿÿÿÿÿÿÿÿ
)
ca# 
	inputs/caÿÿÿÿÿÿÿÿÿ
-
chol%"
inputs/cholÿÿÿÿÿÿÿÿÿ
)
cp# 
	inputs/cpÿÿÿÿÿÿÿÿÿ
/
exang&#
inputs/exangÿÿÿÿÿÿÿÿÿ
+
fbs$!

inputs/fbsÿÿÿÿÿÿÿÿÿ
3
oldpeak(%
inputs/oldpeakÿÿÿÿÿÿÿÿÿ
3
restecg(%
inputs/restecgÿÿÿÿÿÿÿÿÿ
+
sex$!

inputs/sexÿÿÿÿÿÿÿÿÿ
/
slope&#
inputs/slopeÿÿÿÿÿÿÿÿÿ
-
thal%"
inputs/thalÿÿÿÿÿÿÿÿÿ
3
thalach(%
inputs/thalachÿÿÿÿÿÿÿÿÿ
5
trestbps)&
inputs/trestbpsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿà
$__inference_model_layer_call_fn_4374·$%¢
¢
÷ªó
+
age$!

inputs/ageÿÿÿÿÿÿÿÿÿ
)
ca# 
	inputs/caÿÿÿÿÿÿÿÿÿ
-
chol%"
inputs/cholÿÿÿÿÿÿÿÿÿ
)
cp# 
	inputs/cpÿÿÿÿÿÿÿÿÿ
/
exang&#
inputs/exangÿÿÿÿÿÿÿÿÿ
+
fbs$!

inputs/fbsÿÿÿÿÿÿÿÿÿ
3
oldpeak(%
inputs/oldpeakÿÿÿÿÿÿÿÿÿ
3
restecg(%
inputs/restecgÿÿÿÿÿÿÿÿÿ
+
sex$!

inputs/sexÿÿÿÿÿÿÿÿÿ
/
slope&#
inputs/slopeÿÿÿÿÿÿÿÿÿ
-
thal%"
inputs/thalÿÿÿÿÿÿÿÿÿ
3
thalach(%
inputs/thalachÿÿÿÿÿÿÿÿÿ
5
trestbps)&
inputs/trestbpsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
"__inference_signature_wrapper_4316ä$%¨¢¤
¢ 
ª
$
age
ageÿÿÿÿÿÿÿÿÿ
"
ca
caÿÿÿÿÿÿÿÿÿ
&
chol
cholÿÿÿÿÿÿÿÿÿ
"
cp
cpÿÿÿÿÿÿÿÿÿ
(
exang
exangÿÿÿÿÿÿÿÿÿ
$
fbs
fbsÿÿÿÿÿÿÿÿÿ
,
oldpeak!
oldpeakÿÿÿÿÿÿÿÿÿ
,
restecg!
restecgÿÿÿÿÿÿÿÿÿ
$
sex
sexÿÿÿÿÿÿÿÿÿ
(
slope
slopeÿÿÿÿÿÿÿÿÿ
&
thal
thalÿÿÿÿÿÿÿÿÿ
,
thalach!
thalachÿÿÿÿÿÿÿÿÿ
.
trestbps"
trestbpsÿÿÿÿÿÿÿÿÿ"/ª,
*
target 
targetÿÿÿÿÿÿÿÿÿ 
@__inference_target_layer_call_and_return_conditional_losses_4674\$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 x
%__inference_target_layer_call_fn_4663O$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ