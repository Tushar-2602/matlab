exp1: 
a. unit sample sequence
b. unit step sequence
c. unit ramp sequence
d. exponential sequence

% Code for Unit Sample Sequence
y = -2:1:2;
x = [0 0 1 0 0];
stem(y, x);
title('Unit Sample Sequence');
xlabel('y');
ylabel('x');
figure;

% Code for Unit Step Sequence
y = -2:1:2;
x = [0 0 1 1 1];
stem(y, x);
title('Unit Step Sequence');
xlabel('y');
ylabel('x');
figure;

% Code for Ramp Sequence
y = -4:1:4;
x = -4:1:4;
stem(y, x);
title('Ramp Sequence');
xlabel('y');
ylabel('x');
figure;

% Code for Exponential Sequence
y = -4:1:4;
x = exp(y);
stem(y, x);
title('Exponential Sequence');
xlabel('y');
ylabel('x');

exp2:
Aliasing effect

% Code (Aliasing)
f1 = 100;
f2 = 50;
f3 = 20;
fs = 300;
n = [0:1/fs:1];

x1 = 2 * pi * n * f1;
x2 = 2 * pi * n * f2;
x3 = 2 * pi * n * f3;

y1 = sin(x1);
y2 = sin(x2);
y3 = sin(x3);

s = y1 + y2 + y3;
z = abs(fft(s));

figure;
stem(z);
title('Aliasing Plot');
xlabel('Frequency');
ylabel('Amplitude');

figure;
plot(x1, y1);
title('Signal y1');
xlabel('Time');
ylabel('Amplitude');

figure;
plot(x2, y2);
title('Signal y2');
xlabel('Time');
ylabel('Amplitude');

figure;
plot(x3, y3);
title('Signal y3');
xlabel('Time');
ylabel('Amplitude');

% Code (True Sampling)
f1 = 100;
f2 = 50;
f3 = 20;
fs = 200;
n = 0:1/fs:1;

x1 = 2 * pi * n * f1;
x2 = 2 * pi * n * f2;
x3 = 2 * pi * n * f3;

y1 = sin(x1);
y2 = sin(x2);
y3 = sin(x3);

s = y1 + y2 + y3;
z = abs(fft(s));

figure;
stem(z);
title('True Sampling Plot');
xlabel('Frequency');
ylabel('Amplitude');

% Code (for fs > 2*fm)
f1 = 100;
f2 = 50;
f3 = 20;
fs = 300;
n = 0:1/fs:1;

x1 = 2 * pi * n * f1;
x2 = 2 * pi * n * f2;
x3 = 2 * pi * n * f3;

y1 = sin(x1);
y2 = sin(x2);
y3 = sin(x3);

s = y1 + y2 + y3;
z = abs(fft(s));

figure;
stem(z);
title('Plot for fs > 2*fm');
xlabel('Frequency');
ylabel('Amplitude');

exp3:
a. Linearity
b. Time invariance


clc;
clear all;

x1 = [1 2 3 4 5];
h = [0 1 2 3 7];

% Convolution of x1 with h
y1 = conv(x1, h);

x2 = [3 4 5 6 7];

% Combination of signals
x = 2 * x1 + 3 * x2;

% Convolution of the combined signal with h
y = conv(x, h);

% Linear combination of individual convolutions
z = 2 * y1 + 5 * conv(x2, h);

% Plot the results
figure;
plot(y);
title('Convolution of combined signal x with h');
figure;
plot(z);
title('Linear combination of convolutions');

clc;

% Time vector
t = linspace(-1, 1, 1000);

% Rectangular function
rect = double(abs(t) <= 0.5);

% Fourier Transform of the rectangular function
f_transform = fftshift(fft(rect));

% Frequency vector
f = linspace(-500, 500, length(f_transform));

% Normalization of the Fourier transform
f_transform = f_transform / max(abs(f_transform));

% Plotting the rectangular function
figure;
subplot(2, 1, 1);
plot(t, rect);
title('Rectangular function');
xlabel('Time (t)');
ylabel('Amplitude');
grid on;

% Plotting the magnitude of the Fourier Transform
subplot(2, 1, 2);
plot(f, abs(f_transform));
title('Magnitude of Fourier Transform');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

% Frequency-shifted signal
rect2 = exp(1i * 5 * t) .* rect;

% Fourier Transform of the frequency-shifted signal
y2 = fft(rect2);

% Plotting the frequency-shifted signal's Fourier Transform
figure;
plot(f, fftshift(abs(y2)));
title('Frequency shifting');
xlabel('Frequency');
ylabel('Amplitude');
grid on;

exp4:
linear sequence convolution

x = [1, 2, 3, 4, 5];
h = [3, 4, 5, 6, 7];
y = conv(x, h);
n = 1:(length(x) + length(h) - 1);

stem(n, y, 'filled');
title('Using convolution method');
xlabel('Index');
ylabel('Value');


figure;
hold on;

x = [1, 2, 3, 4, 5];
h = [3, 4, 5, 6, 7];
i=1
j=1
k=1
product=0
while i<=9
    temj=j
    temk=k
    if temj >5
        temj=5
        k=k+1
        temk=k
    end
    while temj>=1 && temj<=length(x) && temk<=5
        product=product+x(temj)*h(temk)
        temj=temj-1
        temk=temk+1
    end
    stem(i,product,'filled')
    i=i+1
    j=j+1
 
    product=0
end
title('from loop')
xlabel('index')
ylabel('value')
grid on
hold off

exp5:
autocorrelation and cross correlation

function reversedArray = reverseArray(arr)
    reversedArray = arr(end:-1:1);
end
originalArray = [1, 2, 3, 4, 5];
reversedArray = reverseArray(originalArray);
%doing using xcorr
c=xcorr(originalArray)
figure;
stem(c);
title('Autocorrelation using xcorr of x');
temparray = zeros(1, 9);
n1=length(originalArray)
n2=length(reversedArray)
N=n1+n2-1;
y=zeros(1,N);
for i=1:n1
    for j=1:n2
        y(i+j-1)=y(i+j-1)+originalArray(i)*reversedArray(j)
    end
end
figure
stem(y,'filled')
title('Autocorrelation using loop')
xlabel('lag')
ylabel('amplitude')
grid on
originalArray2=[5 ,6 ,7 ,8 ,9];
reversedArray2 = reverseArray(originalArray2);
%doing cross correlation using xcorr
d=xcorr(originalArray,originalArray2)
figure
stem(d)
title('cross correlation using xcorr(x,y)')
xlabel('lag')
ylabel('amplitude')
temparray2=zeros(1,N);
y2=zeros(1,N);
for i=1:n2
    for j=1:n1
        y2(i+j-1)=y2(i+j-1)+originalArray(j)*reversedArray2(i)
    end
end
figure
stem(y2,'filled')
title('Cross-correlation using loop')
xlabel('lag')
ylabel('amplitude')
grid on



exp6:
inverse z transform

% Given numerator and denominator
numerator=[1,0.3]
denominator=[1,-0.5,0.2,-0.1]


% Find zeros, poles, and residues
[residues, poles, direct_term] = residuez(numerator, denominator);
zeros_of_Hz = roots(numerator);
poles_of_Hz = roots(denominator);

% Display zeros, poles, and residues
disp('Zeros of H(z):');
disp(zeros_of_Hz);
disp('Poles of H(z):');
disp(poles_of_Hz);
disp('Residues of H(z):');
disp(residues);

% Plot the pole-zero plot
figure;
zplane(numerator, denominator);
title('Pole-Zero Plot of H(z)');
grid on;

% Check stability (poles inside the unit circle)
disp('Stability Check:');
stable = true;
for i = 1:length(poles)
    if abs(poles(i)) >= 1
        stable = false;
        fprintf('Pole %.2f is outside or on the unit circle. Unstable.\n', poles(i));
    else
        fprintf('Pole %.2f is inside the unit circle.\n', poles(i));
    end
end
if stable
    disp('The system is stable.');
else
    disp('The system is unstable.');
end

% Display the inverse Z-transform sequence in symbolic form
disp('Generated sequence h(n):');
for i = 1:length(residues)
    fprintf('%.2f * (%.2f)^n * u(n)', residues(i), poles(i));
    if i < length(residues)
        fprintf(' + ');
    end
end
if ~isempty(direct_term)
    fprintf(' + %.2f * u(n)\n', direct_term);
else
    fprintf('\n');
end

exp7:
circular convolution

% Given sequences
x = [3, 4, 6, 2];
h = [4, 2, 3, 1];

% Length of sequences (assuming same length for circular convolution)
N = length(x);

% Initialize the result
y = zeros(1, N);

% Perform periodic convolution using a loop
for n = 1:N
    for k = 1:N
        idx = mod(n-k, N) + 1; % Indexing with periodic boundary
        y(n) = y(n) + x(k) * h(idx);
    end
end

% Display the result of the periodic convolution
disp('Periodic convolution using loop:');
disp(y);

% Verify using the cconv command
y_cconv = cconv(x, h, N);

disp('Periodic convolution using cconv:');
disp(y_cconv);

% Check if both results are the same
if isequal(y, y_cconv)
    disp('The two results match!');
else
    disp('The two results do not match.');
end

% Plot the results
figure;

subplot(2, 1, 1);
stem(0:N-1, y, 'filled');
title('Periodic Convolution using Loop');
xlabel('n');
ylabel('Amplitude');
grid on;

subplot(2, 1, 2);
stem(0:N-1, y_cconv, 'filled');
title('Periodic Convolution using cconv');
xlabel('n');
ylabel('Amplitude');
grid on;

exp8:
DFT and FFT

x = [2,4,6,10];  % Input sequence
N = length(x);  % Length of the sequence
X = zeros(1, N);  % Initialize DFT result

% Compute the DFT using loops
for k = 0:N-1
    for n = 0:N-1  % Use 'n' for the inner loop
        X(k+1) = X(k+1) + x(n+1) * exp(-1i * 2 * pi * k * n / N);  % Fixed indexing
    end
end

% Verify using fft function
X_fft = fft(x);  % Compute DFT using fft

% Display results
disp('Manual DFT using loops:');
disp(X);
disp('DFT using fft function:');
disp(X_fft);

% Plot the results
figure;

% Plot real part of DFT
subplot(2, 2, 1);
stem(0:N-1, real(X), 'filled');
title('Real Part of DFT (Manual)');
xlabel('Frequency index');
ylabel('Amplitude');
grid on;

% Plot imaginary part of DFT
subplot(2, 2, 2);
stem(0:N-1, imag(X), 'filled');
title('Imaginary Part of DFT (Manual)');
xlabel('Frequency index');
ylabel('Amplitude');
grid on;

% Plot magnitude of DFT
subplot(2, 2, 3);
stem(0:N-1, abs(X), 'filled');
title('Magnitude of DFT (Manual)');
xlabel('Frequency index');
ylabel('Magnitude');
grid on;

% Plot phase of DFT
subplot(2, 2, 4);
stem(0:N-1, angle(X), 'filled');
title('Phase of DFT (Manual)');
xlabel('Frequency index');
ylabel('Phase (radians)');
grid on;

% Compare results between manual DFT and fft
figure;
subplot(2, 1, 1);
stem(0:N-1, real(X_fft), 'filled');
hold on;
stem(0:N-1, real(X), 'r--'); % Overlay manual DFT result
title('Comparison of Real Parts (Manual vs FFT)');
xlabel('Frequency index');
ylabel('Amplitude');
legend('FFT', 'Manual DFT');
grid on;

subplot(2, 1, 2);
stem(0:N-1, imag(X_fft), 'filled');
hold on;
stem(0:N-1, imag(X), 'r--'); % Overlay manual DFT result
title('Comparison of Imaginary Parts (Manual vs FFT)');
xlabel('Frequency index');
ylabel('Amplitude');
legend('FFT', 'Manual DFT');
grid on;

exp9:
(hard)

exp10:
digital filter

% Initialize the input vector
x = [2 5 8 3];
n = length(x);

% Create an empty matrix
M = zeros(n, n);

% Fill the matrix
for i = 1:n
    M(i, i:end) = x(1, 1:n-i+1);
end

% Define the vector b
b = [2 1 3 2];

% Multiply b by M
y = b * M;

% Plot y using stem
figure
stem(y, 'filled');

% Perform convolution of x and b
z = conv(x, b);

% Plot the convolution result
figure
stem(z, 'filled');

% Extend x with zeros
x = [x, zeros(1, 3)];
n1 = length(x);

% Create a new matrix with zeros
M = zeros(n1, n1);

% Fill the new matrix
for i = 1:n1
    M(i, i:end) = x(1, 1:n1-i+1);
end

% Extend b with zeros
b = [b, zeros(1, 3)];

% Multiply b with the new matrix
a = b * M;

% Plot the final result
figure
stem(a, 'filled');

exp11:
% dicimation 

x=[1 2 4 5 6]
y_deci=x(1:2:end)
y_inter=upsample(x,2)
y_inter=y_inter(1:length(y_inter)-1)
x_fft=fft(x)
y_deci_fft=fft(y_deci)
y_inter_fft=fft(y_inter)
figure
stem(abs(x_fft))
title('original frequency response')
ylabel('amplitude')
xlabel('frequency')
figure
stem(abs(y_deci_fft))
title('decimated frequency response')
ylabel('amplitude')
xlabel('frequency')
figure
stem(abs(y_inter_fft))
title('interpolated frequency response')
ylabel('amplitude')
xlabel('frequency')

