close all
clear all

thrust = load('thrust_gpu.dat');
vexcl  = load('vexcl_1gpu.dat');
vienna = load('viennacl_gpu.dat');
n  = unique(thrust(:,1))';

thrust_avg = [];
vexcl_avg  = [];
vienna_avg = [];

for i = n
    I = find(thrust(:,1) == i);
    time = sum(thrust(I,2)) / length(I);
    thrust_avg = [thrust_avg time];

    I = find(vexcl(:,1) == i);
    time = sum(vexcl(I,2)) / length(I);
    vexcl_avg = [vexcl_avg time];

    I = find(vienna(:,1) == i);
    time = sum(vienna(I,2)) / length(I);
    vienna_avg = [vienna_avg time];
end


figure(1)
set(gca, 'FontSize', 18)


loglog(n, thrust_avg, 'ko-', 'linewidth', 1, 'markersize', 6, 'markerfacecolor', 'w');
hold on
loglog(n, vexcl_avg,  'ro-', 'linewidth', 1, 'markersize', 6, 'markerfacecolor', 'w');
loglog(n, vienna_avg, 'bo-', 'linewidth', 1, 'markersize', 6, 'markerfacecolor', 'w');

xlim([1e2 1e7])
set(gca, 'xtick', [1e2 1e3 1e4 1e5 1e6 1e7])
xlabel('N');
ylabel('T (sec)');

legend('thrust gpu', 'vexcl 1 gpu', 'viennacl gpu', 'location', 'northwest');
legend boxoff

print('-depsc', 'abs.eps');
