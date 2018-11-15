Research papers:
https://projekter.aau.dk/projekter/files/63218113/report.pdf
https://www.irjet.net/archives/V4/i4/IRJET-V4I4275.pdf
http://thesai.org/Downloads/Volume7No1/Paper_93-Traffic_Sign_Detection_and_Recognition.pdf
http://www.scitepress.org/Papers/2018/67188/67188.pdf
https://pdfs.semanticscholar.org/5fd7/eca5432e4e4038dfc2ad425fa5f3e8739f0b.pdf
https://ieeexplore.ieee.org/document/7124941
https://e-archivo.uc3m.es/bitstream/handle/10016/7089/traffic_escalera_IVC_2003_ps.pdf
https://arxiv.org/ftp/arxiv/papers/1707/1707.07411.pdf
https://towardsdatascience.com/what-is-the-best-programming-language-for-machine-learning-a745c156d6b7

Haar Cascade
Code example:
https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
How training works:
https://docs.opencv.org/3.4/dc/d88/tutorial_traincascade.html
Blog post tutorial:
https://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html
Another tutorial (check it out):
http://christopher5106.github.io/computer/vision/2015/10/19/create-an-object-detector.html

===================================================================================================
How to do the cascade training:
===================================================================================================
Step 1
- Create a folder structure like this:

/img
  img1.jpg
  img2.jpg
bg.txt

File bg.txt:
img/img1.jpg
img/img2.jpg

==========================================
==========================================
  Introduction
  Diagnosting imaging has become a very important tool in medicine. It provides a way to map the anatomy of a patient which then can be used for diagnosis, treatment planning or medical studies. Medical imaging is used in various modalities such as magnetic resonance imaging (MRI), digital mammography, ultrasound or computed tomography (CT). As technology evolves more and more medical images are being used in every day medicine and reliable methods and algorithms are needed to process the large amount of data aquired from the patients. These algorithms often provide a way to delineate regions of interests (anatomical structures, organs).
  
  Definitions
  "Image segmentation is the process of partitioning an image into distinctive regions (segments) with similar properties such as intensity, texture, brightness, and contrast." [x]
  
  Due to the fact that there are numerous image scans that need to be processed manual segmentation would take a long time and it would often leads to errors and this is why automated methods are needed to perform segmentation.
  
[x] https://ieeexplore.ieee.org/document/7881449

https://pdfs.semanticscholar.org/d909/6dc6d76148633f168c929e6e7961f563ca78.pdf

Moreover, due to the numerous image scans for each patient and similar intensity of the nearby organs, manual segmentation is a tedious task and is prone to errors. Therefore, reliable automated algorithms should be utilized to delineate the region of interest.

------------------------------
1.) Vision For Test Drives (->)
Phase I. (2 HC on board)
Bosch as driver
Bosch as engineer
//measurement requires a lot of adjustments, plus tagging needs to be done
Phase II. (2 HC on board)
AMS driver
Bosch as engineer
//measurement requires a lot of adjustments, plus tagging needs to be done)
Phase III. (2 HC on board)
AMS as driver
BEKO/AMS as engineer
//measurements are becoming more stable, some intervention is required, plus tagging needs to be done)
Phase IV. (1 HC on board)
AMS as driver
No engineer
//measurements are stable, driver can start the equipment and then do ~100km of driving. Tagging is done offline by AI.

2)
Computer Vision (CV) has been rapidly evolved in the last decade. Today very sophisticated APIs are publicly accessible for image recognition, which help the engineer to do these tasks quickly and efficiently. 
For test drives evaulation, knowing the environment is beneficial, that’s why an engineer does tagging during the drives. This activity can be replaced by AI, with adequate accuracy.
As many drive data could be available in the measurements, the auto-tagging function can go pure image recognition. (e.g.: tagging based on road turn radius, rain-sensor signal, environment temperature etc.)

3) Pyramid
AI
Neural NetWorks
CNN - widespread deep-learning technology for image recognition
faster r-cnn
(+: shows better accuracy compared to other solutions, detects smaller features well; -: slow)

4) Object Recognition - Implementations
Matlab (Computer vision toolbox)
+ Easy to use, many tutorials are available for object det.
+ Commonly used within Bosch
- Limited options compared to Python
- Relatively slow
Python
+ Faster computiong
+ Commonly used
+ Good object det API
+ Better parallelization
- Portability and coding is more challanging
Google APi
+ ready to use service
- slow, expensive, confidentality issues

5) Object rec. software overview
Input video files from measurement camera -> main scrip -> frame extractor (provides frame of the video for the image recognition) -> TensorFlow API: object det api provides per frame results -> main scrip -> result file contains frame number and detected objects

6) Configuring the AI
Config file is needed for:
number of objects to be detected (tunnel, gantry, …)
max objects per frame
thresholds
training config options
and more

7) Training the AI
Training is done throurg the convenient  „LabelImg”
100 frames gives adaquate image recognition performance
Training a new object takes approx 1 hour 
On one frame more objects can belabelled at the same time

8) Resulkts so far
The system is capable of recognizing the following objects on a highway with high accuracy:
Truck
Gantry
Bridge
Tagging a 1.5 hour-long video takes about 2 hours
If we use more training data we can improve the accuracy and robustness of the modell

9) Example pictures

10) Example file

11) Further improvements
Make Auto-Tagging an integral part of the eval tool or the post-processing 
Tagging can be extended into situation recognition by applying vehicle date from the vehicle-bus
Training the modell to detect objects that are very similar to our currently detected objects (to avoid false detections)
Applying plausabilization to output file for avoiding false positives
Turning frame number into time data

--------
GNSS bevezető - ESA könyve
https://gssc.esa.int/navipedia/GNSS_Book/ESA_GNSS-Book_TM-23_Vol_I.pdf
--------










------------------------------------------------------------------------------------------
1) Matlab-overview
- matlab stands for Matrix Laboratory
- This software package is designed specifically to perform operations on matrices
- the package consists of a command interpreter, a command language and a collection of routines which can be exercised by the interpreter
- the commands can ether be entered line-by-line in the command window or they can be collected in a text file to be executed by a single line entered in the command window
- all computations internal to MATLAB are done in double precision

2) MATLAB toolboxes
- Math and Analysis: Optimization, Requirements Management Interface, Statistics, Neural Network, Symbolic/Extended Math, Partial Differential Equations, PLS Toolbox, Mapping, Spline
- Data Aquisition and Import: Data Acquisition, Instrument Control, Excel Link, Portable Graph Object
- Signal and Image Processing: Signal processing, image processing, communications, frequency domain system identification, higher order spectral analysis, system identification, wavelet, filter design
- Control Design: Control System, Fuzzy Logic, Robust Control, Model Predictive Control

3) MATLAB System:
- Language: arrays and matrices, control flow, I/O, data structures, user-defined functions and scripts
- Working Environment: editing, variable management, importing and exporting data, debugging, profiling
- Graphics system: 2D and 3D data visualization, animation and custom GUI development
- Mathematical Funcation: basic (sum, sin) to advanced (fft, inv, Bessel funcation)
- API: can use MATLAB with C, Fortran, and Java in either direction

4) MATLAB desktop and environment
Command Window: where you enter commands
Command History: running history of commands which is  preserved across MATLAB sessions
Current directory: Default is $matlabroot/work
Workspace: GUI for viewing, loading and saving MATLAB variables
Array Editor: GUI for viewing and/or modifying contents of MATLAB variables (openvarvarnameor doubleclick the array’s namein the Workspace)
Editor/Debugger: text editor, debugger; editor works with file types in addition to .m(MATLAB “mfiles”)

5) Environment:
current directory, type commands here, command window, workspace

6) Variables (Arrays) and Operators-Variable Basics
>> 16 + 24
ans = 40    (No declarations needed)
>> product = 16 * 23.24
product = 371.84    (mixed data types)
>> product = 16 * 555.24
>> product
product = 8883.8    (semi-colon suppresses output of the calculation's result)

>> clear  (clear removes all variables) (clear x y removes only x and y)
>> product = 2 * 3^3;
>> comp_sum = (2 + 3i) + (2 - 3i);
>> show_i = i^2;
>> save three_things (save/load are used to retain/restore workspace variables)
>> clear
>> load three_things
>> who
Your variables are:
comp_sum  product   show_i    
>> product
product =
54
>> show_i
show_i =
- 1   (use home to clear screen and put cursor and the top of the screen)

MATLAB Data
The basic data type used in MATLAB is the double precision array
•No declarations needed: MATLAB automatically allocates required memory
•Resize arrays dynamically
•To reuse a variable name, simply use it in the left hand side of an assignment statement
•MATLAB displays results in scientific notation
  o Use File/Preferences and/or format function to change default
    short (5 digits), long (16 digits)
    format short g; format compact (my preference)

Data types
- Array (full or sparse)
  - logical
  - char
  - numeric
    - int8, uint8, int16, uint16, int32, uint32, int64, uint64
    - single
    - double
  - cell
  - structure
    - user classes
  - java classes
  - function handle
  
Variables Revisited
Variable names are case sensitive and  over-written when re-used
Basic variable class: Auto-Indexed Array
  Allows use of entire arrays (scalar, 1-D, 2-D, etc...) as operands
  Vectorization: Always use array operands to get best performance (see next slide)
Terminology: “scalar” (1 x 1 array), “vector” (1 x N array), “matrix” (M x N array)
Special variables/functions: ans, pi, eps, inf, NaN, i, nargin, nargout, varargin, varargout, ...

Special variables
- ans - default variable name for results
- pi - value of pi
- eps - smallest incremental number
- inf - infinity
- NaN - not a number e.q. 0/0
- realmin - the smallest usable positive real number
- realmax - the larges tusable psoitive real number

----
MATLAB Matrices
•
MATLAB treats all variables as matrices. For our 
purposes a matrix can be thought of as an array, 
in fact, that is how it is stored.
•
Vectors are special forms of matrices and 
contain only one row OR one column.
•
Scalars are matrices with only one row AND 
one column
---

Data I/O - loading and saving workspace variables
MATLAB can load and save data in 
.MAT 
format
.MAT
files are binary files that can be transferred across platforms; 
as much accuracy as possible is preserved.

Load:
load filename
OR
A = load(‘filename’)
loads all the variables in the specified file (the default name is
MATLAB.MAT
)

Save:
save filename variables
saves the specified variables (
all
variables by default) in the specified
file (the default name is
MATLAB.MAT
)
-.------
When using MATLAB, you may wish to leave the program but save the 
vectors and matrices you have defined. 

SAVE
, Save workspace variables to disk.
SAVE FILENAME
saves all workspace variables to the binary "
MAT
-
file
" named 
FILENAME.mat
. 

The data may be retrieved with 
LOAD
. 

If 
FILENAME
has no extension, 
.mat
is assumed. 
SAVE
, by itself, creates the 
binary
"
MAT
-
file
" named '
matlab.mat
'. 

It is an error if '
matlab.mat
' is not writable. 

To save the file to the working directory, type

>>save filename

SAVE FILENAME X
saves only 
X
. 

SAVE FILENAME X Y Z
saves 
X, Y,
and 
Z
.
where "filename" is a name of your choice. To retrieve the data later, 
type. 

-----------
LOAD
Load workspace variables from disk. 

LOAD FILENAME
retrieves all variables from a file given a full pathname 
or a 
MATLABPATH
relative partial pathname (see PARTIALPATH). 

If FILENAME has no extension 
LOAD 
looks for FILENAME and 
FILENAME.mat
and treats it as a 
binary
"
MAT
-
file
". 

If FILENAME has an extension other than 
.mat
, it is treated as 
ASCII
.
LOAD
, by itself, uses the binary "
MAT
-
file
" named '
matlab.mat
'. It is an 
error if '
matlab.mat
' is not found.
LOAD FILENAME X
loads only 
X
.
LOAD FILENAME X Y Z
... loads just the specified variables. 

>>load x, y, z

See help save and help load for more information.. 

--------
The following describes the use of M
-
files on a PC version of
MATLAB. 

MATLAB requires that the M
-
file must be stored either in the working 
directory or in a directory that is specified in the MATLAB path list. 

For example, consider using MATLAB on a PC with a user
-
defined M
-
file stored in a directory called "
\
MATLAB
\
MFILES
";. 

Then to access that M
-
file, either change the working directory by 
typing 
cd
\
matlab
\
mfiles
from within the MATLAB command window or 
by adding the directory to the path. 

Permanent addition to the path is accomplished by editing the 
\
MATLAB
\
matlabrc.m
file. 

Temporary modification to the path is accomplished by typing 
path(path,'
\
matlab
\
mfiles')
from within MATLAB. 
-----------------

Customize MATLAB’s start
-
up behaviour

Create 
startup.m
file and place in:

Windows
: 
$matlabroot
\
work

UNIX
: directory where 
matlab
command is issued
My
startup.m
file:
addpath e:
\
download
\
MatlabMPI
\
src
addpath e:
\
download
\
MatlabMPI
\
examples
addpath .
\
MatMPI
format short g

----

ype “help” at the command prompt
and MATLAB returns a list of help topics
-----

keyword search of help entries
>> lookfor who
WHO    List current variables.
WHOS List current variables, long form. 
>> 
who
Your variables are:
ans
fid I
>> whos
Name    Size   Bytes  Class    Attributes
ans     1x1        8  double              
fid     1x1        8  double              
i       1x1        8  double 
-------
you can access “on
-
line” help by clicking the
question mark in the toolbar

-----
Summary
•
help 
command

Online help
•
lookfor 
keyword

Lists related commands
•
which

Version and location info
•
clear

Clears the workspace
•
clc

Clears the command window
•
diary 
filename

Sends output to file
•
diary on/off

Turns diary on/off
•
who, whos

Lists content of the workspace
•
more on/off

Enables/disables paged output
•
Ctrl+c

Aborts operation
•
...

Continuation
•
%

Comments
----------
Plotting Elementary Functions: 
MATLAB supports many types of graph and surface 
plots: 
2 dimensions line plots (x vs. y), filled plots, bar 
charts, pie charts, parametric plots, polar plots, 
contour plots, density plots, log axis plots, surface 
plots, parametric plots in 3 dimensions and 
spherical plots. 
-----------
2
-
D
plots
:
The
plot
command
creates
linear
x
-
y
plots
;
if
x
and
y
are
vectors
of
the
same
length,
the
command
plot(x,y)
opens
a
graphics
window
and
draws
an
x
-
y
plot
of
the
elements
of
x
versus
the
elements
of
y
.
-------
>>t=
-
1:0.01:1; 
>>f=4.5*cos(2*pi*t 
-
pi/6); 
>>%The following statements plot the sequence and label the plot 
>>plot(t,f),title('Fig.E1.2a'); 
>>axis([
-
1,1,
-
6,6]); 
>>xlabel('t'); 
>>ylabel('f(t)'); 
>>text(
-
0.6,5,'f(t) = A cos(wt + phi)'); 
>>grid;

----------
PLOT(X,Y)
plots vector Y versus vector X.
TITLE('text')
adds text at the top of the current plot. 
XLABEL('text')
adds text beside the X
-
axis on the current axis. 
YLABEL('text')
adds text beside the Y
-
axis on the current axis.
GRID,
by itself, toggles the major grid lines of the 
current axes.
GTEXT('string')
displays the graph window, puts up a cross
-
hair,    and 
waits for a mouse button or keyboard key to be pressed.
SUBPLOT(m,n,p),
or SUBPLOT(mnp), breaks the Figure window into an m
-
by
-
n matrix of small axes.
STEM(Y)
plots the data sequence Y as stems from the x axis 
terminated with circles for the data value.
SEMILOGX(...)
is the same as PLOT(...), except a logarithmic (base 10) 
scale is used for the X
-
axis.
SEMILOGY(...)
is the same as PLOT(...), except a logarithmic (base 10) 
scale is used for the Y
-
axis.. 
------------
By default, the axes are auto
-
scaled. 
This can be overridden by the command 
axis
. If 
c = 
[xmin,xmax,ymin,ymax]
is a 
4
-
element vector, then 
axis(
c
)
sets the axis scaling 
to the prescribed limits. 
By itself, axis freezes the current scaling for 
subsequent graphs; entering axis again returns to 
auto
-
scaling. 
The command 
axis('square')
ensures that the same 
scale is used on both axes. 
For more information's on axis see help axis. . 

--------
Two ways to make multiple plots on a single graph are illustrated by
•
>>t = 0:.01:2*pi; 
•
>>y1 = sin(t); y2=sin(2*t); y3=sin(4*t) 
•
>>plot(t,y1,y2,y3) 
•
and by forming a matrix Y containing the functional values as columns 
•
>>t = 0:.01:2*pi; 
•
>>y = [sin(t)', sin(2*t)', sin(4*t)'] 
•
>>plot(t,y) 
•
Another way is with the hold command. The command hold freezes the 
current graphics screen so that subsequent plots are superimposed on it. 
Entering hold again releases the "hold". The commands hold on and hold off 
are also available. 
•
One can override the default linotypes and point types. For example, 
•
>>t = 0:.01:2*pi; 
•
>>y1 = sin(t); y2=sin(2*t); y3=sin(4*t) 
•
>>plot(t,y1,'
--
',y2,':',y3,'+')
-
------------

40
Internal  | Department  | 
04
/
04
/
2012 
| RBEI
-
NE
1 
| ©  Robert Bosch Engineering and Business Solutions Limited 
2012
. All rights reserved, also 
regarding any disposal, exploitation, reproduction, editing, distribution, as well as in the event of applications for indust
ria
l property rights.
Plotting Elementary Functions: 
•
Colors                                Line Styles 
•
y
yellow                               
.
point
•
M 
magenta                           
o 
circle
•
C
cyan                                  
x 
x
-
mark
•
R
red                                    
+
plus 
•
G
green                                
-
solid
•
B 
blue                                   
*
star
•
W
white                                
:
dotted
•
K
black                                
-
.
Dashdot
•
--
dashed
More mark types are; 
square(s), diamond(d), up
-
triangle(v), down
-
triangle(^), left
-
triangle(<), right
-
triangle(>), pentagram(p), 
hexagram(h)
See also help plot for more line and mark color
-----------
The command subplot can be used to partition the screen so that up to 
four plots can be viewed simultaneously. See 
help subplot
. 
Example for use of subplot: 
>>% Line plot of a chirp 
>> x=0:0.05:5; 
>> y=sin(x.^2); 
>> subplot(2,2,1), plot(x,y); 
>> % Bar plot of a bell shaped curve 
>> x = 
-
2.9:0.2:2.9; 
>> subplot(2,2,2), bar(x,exp(
-
x.*x)); 
>> % Stem plot 
>> x = 0:0.1:4; 
>> subplot(2,2,3), stem(x,y) 
>> % Polar plot 
>> t=0:.01:2*pi; 
>> subplot(2,2,4), polar(t,abs(sin(2*t).*cos(2*t)));
----------
script file to generate a 
graph of y = sin(t)
----------
function file to generate a 
graph of y = sin(t)
>>graphsin
>>
-----------
“legend” remembers the 
order the graphs were 
plotted
---------
MATLAB BASICS
Vector
A vector is defined as a combination of variables  values to with 
components of xj 
, where j = 1,...n values.
-----------
creating vector
- row separator: space/coma
  a = [1 2 3]
  a = [1, 2, 3]
- creating sequences:
  from : jump : till
  linespec(X1, X2, N) generates n point between x1 and x2
    a3=1:1:3
 - column separator: semicolon(;)
    a4 = [1, 2, 3]
-------
A matrix is a two dimensional arrays, where the matrix B is represented 
by a [ m x n ]
--------

Creating Matrix

>> A=[
1 2 3 
; 
5 6 7 
; 
8 9 10
]
A =

1     2     3
5     6     7
8     9    10

>>
B=[
1 2 
; 
3 4 
; 
5 6 
;]
B =
1     2
3     4
5     6

-----------
Colon Operator (Vector Creation)
>> 1:5 % use the colon operator to 
create row vectors
ans =
1     2     3     4     5
>> 1:
0.9
:6 % you can 
vary the increment
(0.9 in this case)
ans =
1.0000   1.9000   2.8000   3.7000   4.6000   5.5000
The last element is always 
less than or equal to
the upper limit
-----------
Colon Operator (Indexing)
>> sum(A(
1:3
,3)) % sums first three 
% elements of column 3
ans =
20
>> sum(A(:,
end
)) % a lone colon is 
ALL
% elements, 
end
is 
% the 
last 
element
ans =
20
-------------

Vector element Operations

Individual addition 
A + B
A + B

Individual subtraction
A 
–
B
A 
-
B

Individual multiplication
A*B
A.*B

Individual division (left)
A/B
A./B

Individual division (right) 
A
\
B
A.
\
B

Individual power
A
B
A.^B
----------------
The “Dot Operator”

By default and whenever possible 
MATLAB will perform true
matrix operations (
+
-
*
).
The operands in every arithmetic 
expression are considered to be matrices
.

If, on the other hand, the user wants the 
scalar  version
of an 
operation a “dot” must be put in front of the operator, e.g., 
.*
. 
Matrices can still be the operands but the mathematical calculations
will be performed 
element
-
by
-
element
.

A comparison of matrix multiplication and scalar multiplication
is shown on the next slide.
------------
Addition
Subtraction
Product
Transpose
---------
ector Multiplication
>>a=[2 3]
>>b=[3 2]
>>a*b
>>a.*b
>>a.*b’
>>a*b’
-----------
Matrix Multiplication

Inner dimensions must be equal

Dimension of resulting matrix = outermost
dimensions of multiplied matrices

Resulting elements = dot product of the rows of
the 1st matrix with the columns of the 2nd matrix
----------
Dot Operator Example
>> A = [
1 5 6
; 
11 9 8
; 
2 34 78
]
A =
1     5     6
11     9     8
2    34    78
>> B = [
16 4 23
; 
8 123 86
; 
67 259 5
]
B =
16     4    23
8   123    86
67   259     5
-----------

Dot Operator Example (cont.)
>> C = A * B   % “normal” matrix multiply
C =
458        2173         483
784        3223        1067
5530       24392        3360
>> CDOT =
A
.*
B
% 
element
-
by
-
element
CDOT =
16          20         138
88        1107         688
134        8806         390
-----------------
Two Division Operators

Right divide (familiar version)   
a/b

What happens: 
a
is divided by 
b

Right  operand “goes into” left operand

Left divide
a
\
b

What happens: 
b
is divided by 
a

Left operand “goes into” right operand

Behaviour depends on operands (scalar vs. matrix)

Both operators work with matrices (of course). More later on 
what is actually calculated ...
Comparison of the use of 
/
and 
\
on next slide
-------------
Using the Division Operators
>> x = 53.0;
>> y = 22.5;
>> x/y
ans = 2.3556
>> x
\
y
ans = 0.4245
>> (x/y)^(
-
1)
ans = 0.4245
-------------
Extracting column and rows from matrix
>> A=[1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
A =
1     2     3     4
5     6     7     8
9    10    11    12
13    14    15    16
1.Exract an element of a matrix:
>> A(2,3)
ans =
7
-------------
64
Internal  | Department  | 04/04/2012 | RBEI
-
NE1 | ©  Robert Bosch Engineering and Business Solutions Limited 2012. All rights re
served, also 
regarding any disposal, exploitation, reproduction, editing, distribution, as well as in the event of applications for indust
ria
l property rights.
MATLAB BASICS
MATLAB BASICS
Extracting column and rows from matrix
2. Extracting a column
>> A(:,3)
ans =
3
7
11
15
3. Extracting a row
>> A(3,:)
ans =
9    10    11    12
4. Part of matrix
>> A(2:4,2:3)
ans =
6     7
10    11
14    15
----------------------
Other matrix operation
1 rand - generates random numbers
rand(2,3)
2 magic - generates magic square
magic(3)ú
3 eye
4 ones
5 zeroes
6 transpose of a matrix
7 horizontal concatination
  A = [1 2 3]
  B = [5 6 7]
  c = [A, B]
8 vertical concatination
  c=[A;B]
  -----------
  
Scripts and Functions

Scripts
do not accept input arguments, nor do they produce 
output arguments.  Scripts are simply MATLAB commands 
written into a file. They operate on the 
existing workspace
. 

Functions
accept input arguments and produce output variables.
All internal variables are local to the function and commands 
operate on the 
function workspace
.

A file containing a script or function is called an
m
-
file

If duplicate functions (names) exist, the first in the search path 
(from 
path
command) is executed.
--------
Functions

Functions describe subprograms

Take inputs, generate outputs

Have local variables (invisible in global workspace)

Core MATLAB (Built
-
in) Functions

sin, abs, exp, ...
Can’t be displayed on screen

MATLAB
-
supplied M
-
file Functions

mean, linspace, ...
Ca be displayed on screen

User
-
created M
-
file Functions
------------------------

MATLAB BASICS
- Core MATLAB (Built-in) Functions
  - elementary built-in functions
  >> help elfun 
% a list of these functions
sin
% Sine.
exp
%
Exponential.
abs
%
Absolute value. 
round
%
Round towards nearest integer

Special Math functions 
lcm
% Least common multiple.
cart2sph
% Transform Cartesian to spherical
%
coordinates.

Special functions 
-
toolboxes 
Each toolbox has a list of special functions that you can use

----------------
Structure of a Function M-file

function y = mean(x)  (keyword: function, y: output arguments, mean: function name same as file name, x: input arguments)
[m,n] = size(x)
if m == 1
  m = n;
end
y = sum(x)/m

usage
output_value = mean(input_value
----------

Multiple Input & Output Arguments
function
r = ourrank(X,tol)
% OURRANK Rank of a matrix
s = svd(X);
if
(nargin == 
1
)
tol = max(size(X))*s(
1
)*eps;
end
r = sum(s > tol);

----
function
[mean,stdev] = ourstat(x)
% OURSTAT Mean & std. deviation
[m,n] = size(x);
if
m == 1
m = n;
end
mean = sum(x)/m;
stdev = sqrt(sum(x.^2)/m 
–
mean.^2);

--------
nargin 
–
number of input arguments
-
Many of MATLAB functions can be run with different number of 
input variables. 

nargout 
–
number of output arguments
-
efficiency

nargchk 
–
check if number of input arguments is between some ‘low’ 
and ‘high’ values
-----------------

MATLAB Calling Priority
High
variable
built
-
in function
subfunction
private function
MEX-file
P-file
M-file 
Low

----

Reading excel
[num,txt,raw] = xlsread(filename,sheet,range)
>> [num,txt,raw]=xlsread('first_excel.xls','sheet1','A4:D6')

Writing to excel (both *.xls and *.xlsx)
xlswrite(filename,A,sheet,range)
Y=[10;11;12];
>> xlswrite('first_excel.xls',Y,'sheet1','A4:A6')
----------------------




Feladatok
	Adj legalább két(-három)féle megoldást a következő mátrix létrehozására:
A=[■(1&1&1&1&1@5&5&5&5&5@4&4&4&4&4)]
	Egy szkript az alábbi kódot tartalmazza. Milyen hibát (melyik sorban) fog jelezni a Matlab a következő kód futtatásakor?
a = 10*ones(4,1);
b = (a-8).^2+2;
c = [1;2;3;4;5];
d = b/3 * c';
e = sum(c)/det(d);
f = e^2*c; 
	Az alábbi kód futtatásakor mi fog megjelenni a „Command Window”-ban?
function rootfcn
x = 4
y = 3
z = [2 4 5 3 1];
for i = 1:2:5
    if (i <= 2)
        back1 = fcn1(i,x,y,z)
    elseif (i > 3)
        disp('done')
    else
        back2 = fcn2(i,y,x,z)
    end
end

function [out2] = fcn1(j,y,x,k)
out2 = [0 0 0 0 0]
for i=1:3:5
    out2(i) = fcn2(i,x,y,k);
end

function [out1] = fcn2(j,d,s,k)
switch (k(j))
    case {1,2}
        out1 = d+9
    case {3,4}
        out1 = 2*s
    otherwise
        out1 = d+s
end 

Megoldások
	Megoldási opciók:
	közvetlen megadás:
A = [1 1 1 1 1;5 5 5 5 5;4 4 4 4 4];
	vektor konkatenáció:
a = [1;5;4];
A = [a a a a a]; VAGY A = [ones(1,5);5* ones(1,5);4* ones(1,5)];
	mátrix szorzás:
A = [1;5;4]*ones(1,5);
	repmat:
A = repmat([1;5;4],1,5);
	Hiba: 4. sor: determináns számítás nem kvadratikus mátrix-szal: det(d)
	Output:
	sor	x = 4
	sor	y = 3
	sor	out2 = 0 0 0 0 0
	sor	out1 = 12
	sor	out1 = 8
	sor	back1 = 12 0 0 8 0
	sor	out1 = 7
	sor	back2 = 7
	sor	done



