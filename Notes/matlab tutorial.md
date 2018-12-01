## Octave/Matlab Tutorial

### Basic Operation

### Moving Data Around

```matlab
save *.txt <var> -ascii
```

### Computing on Data

```matlab
find()
prod()
floor()
ceil()
flipud()
```

```matlab
max(A, [], 1) % 1: first dimension
max(max(A)) == max(A(:))
sum(sum(A .* eye(9))) % sum of diagnol elements
```

### Plotting Data

```matlab
figure(1)
plot(x, y)
xlabel("x")
ylabel("y")
legend("")
title("")
print -dpng 'myPlot.png'
clf
close
```

```matlab
subplot(1,2,1) % Divide plot a 1x2 gird, access first element
axis([xmin xmax ymin ymax])
```

```matlab
% visualize matrix
imagesc(A)
colorbar
colormap gray
```

### Vectorization

