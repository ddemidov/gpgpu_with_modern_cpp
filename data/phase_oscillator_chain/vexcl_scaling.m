close all
clear all

ms = 6;
lw = 0.5;
fs = 26;

ex = [ ...
    experiment('vexcl_1gpu',        'Tesla  \times 1',  'k:',  'w', ms, lw) ...
    experiment('vexcl_2gpu',        'Tesla  \times 2',  'ko-', 'w', ms, lw) ...
    experiment('vexcl_3gpu',        'Tesla  \times 3',  'kd-', 'w', ms, lw) ...
    experiment('vexcl_1gpu_tahiti', 'Tahiti  \times 1', 'k:',  'k', ms, lw) ...
    experiment('vexcl_2gpu_tahiti', 'Tahiti  \times 2', 'ko-', 'k', ms, lw) ...
    experiment('vexcl_3gpu_tahiti', 'Tahiti  \times 3', 'kd-', 'k', ms, lw) ...
    ];

figure(1)
set(gca, 'FontSize', fs);

for i = 1:length(ex)
    if i == 1 || i == 4
	ref = ex(i).t;
    end

    ex(i).semilogx(ref ./ ex(i).t);
    hold on
end

xlim([1e2 1e7])
ylim([0 3])
set(gca, 'xtick', [1e2 1e3 1e4 1e5 1e6 1e7])
xlabel('N');
ylabel('T(1 GPU) / T');
legend(ex(:).legend, 'location', 'NorthWest');
legend boxoff
axis square

print('-depsc', 'scaling.eps');
