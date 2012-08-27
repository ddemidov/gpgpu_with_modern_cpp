close all
clear all

t  = load('thrust_gpu.dat');
tc = load('thrust_cpu.dat');
v1 = load('vexcl_1gpu.dat');
v2 = load('vexcl_2gpu.dat');
v3 = load('vexcl_3gpu.dat');
vc = load('vexcl_cpu.dat');
n  = unique(t(:,1))';

tavg  = [];
tcavg = [];
v1avg = [];
v2avg = [];
v3avg = [];
vcavg = [];

for i = n
    I = find(t(:,1) == i);
    time = sum(t(I,2)) / length(I);
    tavg = [tavg time];

    I = find(tc(:,1) == i);
    time = sum(tc(I,2)) / length(I);
    tcavg = [tcavg time];

    I = find(v1(:,1) == i);
    time = sum(v1(I,2)) / length(I);
    v1avg = [v1avg time];

    I = find(v2(:,1) == i);
    time = sum(v2(I,2)) / length(I);
    v2avg = [v2avg time];

    I = find(v3(:,1) == i);
    time = sum(v3(I,2)) / length(I);
    v3avg = [v3avg time];

    I = find(vc(:,1) == i);
    time = sum(vc(I,2)) / length(I);
    vcavg = [vcavg time];
end


figure(1)
set(gca, 'FontSize', 18)

loglog(n, tcavg, 'kd-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');

hold on

loglog(n, tavg, 'ko-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');

loglog(n, vcavg, 'md-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');

loglog(n, v1avg, 'ro-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');

loglog(n, v2avg, 'bo-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');

loglog(n, v3avg, 'go-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');

xlim([1e2 1e7])
set(gca, 'xtick', [1e2 1e3 1e4 1e5 1e6 1e7])
xlabel('N');
ylabel('T (sec)');

legend('thrust cpu', 'thrust gpu', 'vexcl cpu', 'vexcl 1 gpu', 'vexcl 2 gpu', 'vexcl 3 gpu', ...
    'location', 'northwest');
legend boxoff

print('-depsc', 'abs.eps');

figure(2)
set(gca, 'FontSize', 18)

loglog(n, tcavg ./ tavg, 'kd-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');

hold on

loglog(n, vcavg ./ tavg, 'md-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');

loglog(n, v1avg ./ tavg, 'ro-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');

loglog(n, v2avg ./ tavg, 'bo-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');

loglog(n, v3avg ./ tavg, 'go-', ...
		'linewidth', 2, 'markersize', 6, 'markerfacecolor', 'w');

loglog(n, ones(size(n)), 'k:');

h = legend('thrust cpu', 'vexcl cpu', 'vexcl 1 gpu', 'vexcl 2 gpu', 'vexcl 3 gpu');
legend boxoff
set(h, 'FontSize', 16)

xlim([1e2 1e7])
ylim([1e-1 1e3])
set(gca, 'xtick', [1e2 1e3 1e4 1e5 1e6 1e7])
xlabel('N');
ylabel('T / T(thrust gpu)');

print('-depsc', 'rel.eps');
