close all
clear all

figure(1)
set(gcf, 'Position', [50, 50, 800, 900]);

ms = 3;
lw = 0.5;
fs = 8;

% CPU
ex = [ ...
    experiment('thrust_cpu',         'Thrust CPU',	    'ko-', 'w', ms, lw) ...
    experiment('viennacl_cpu_intel', 'ViennaCL CPU (Intel)','kv-', 'k', ms, lw) ...
    experiment('viennacl_cpu_amd',   'ViennaCL CPU (AMD)',  'kv-', 'w', ms, lw) ...
    experiment('vexcl_cpu_intel',    'VexCL CPU (Intel)',   'kd-', 'k', ms, lw) ...
    experiment('vexcl_cpu_amd',      'VexCL CPU (AMD)',     'kd-', 'w', ms, lw) ...
    experiment('thrust_gpu',         'Thrust Tesla',	    'k:',  'w', ms, lw) ...
    ];

subplot(2, 2, 1); set(gca, 'FontSize', fs);
subplot(2, 2, 2); set(gca, 'FontSize', fs);

ref = ex(6).t;
for i = 1:length(ex)
    subplot(2, 2, 1);
    ex(i).loglog();
    hold on

    subplot(2, 2, 2);
    ex(i).loglog(ex(i).t ./ ref);
    hold on
end

subplot(2, 2, 1);
xlim([1e2 1e7])
ylim([1e-2 1e7])
set(gca, 'xtick', [1e2 1e3 1e4 1e5 1e6 1e7])
set(gca, 'xticklabel', [])
ylabel('T (sec)');
legend(ex(:).legend, 'location', 'NorthWest');
legend boxoff
axis square

pos1 = get(gca, 'Position');

subplot(2, 2, 2);
xlim([1e2 1e7])
set(gca, 'xtick', [1e2 1e3 1e4 1e5 1e6 1e7])
set(gca, 'xticklabel', [])
set(gca, 'yaxislocation', 'right')
ylabel('T / T(Thrust)');
axis square

pos2 = get(gca, 'Position');
pos2(1) = pos1(1) + pos1(3) - 0.05;
set(gca, 'Position', pos2);

gray = [0.6 0.6 0.6];

% GPU
ex = [ ...
    experiment('thrust_cpu',          'Thrust CPU',	 'k:',  'w', ms, lw) ...
    experiment('thrust_gpu',          'Thrust Tesla',	 'ko-', 'w', ms, lw) ...
    experiment('cmtl4_gpu',           'MTL4 Tesla',	 'ks-', 'w', ms, lw) ...
    experiment('viennacl_gpu',        'ViennaCL Tesla',  'kv-', 'k', ms, lw) ...
    experiment('viennacl_gpu_tahiti', 'ViennaCL Tahiti', 'kv:',gray, ms, lw) ...
    experiment('vexcl_1gpu',          'VexCL Tesla',     'kd-', 'k', ms, lw) ...
    experiment('vexcl_1gpu_tahiti',   'VexCL Tahiti',    'kd:',gray, ms, lw) ...
    ];

subplot(2, 2, 3); set(gca, 'FontSize', fs);
subplot(2, 2, 4); set(gca, 'FontSize', fs);

ref = ex(2).t;
for i = 1:length(ex)
    subplot(2, 2, 3);
    ex(i).loglog();
    hold on

    subplot(2, 2, 4);
    ex(i).loglog(ex(i).t ./ ref);
    hold on
end

subplot(2, 2, 3);
xlim([1e2 1e7])
ylim([1e-1 1e5])
set(gca, 'xtick', [1e2 1e3 1e4 1e5 1e6 1e7])
xlabel('N');
ylabel('T (sec)');
legend(ex(:).legend, 'location', 'NorthWest');
legend boxoff
axis square

pos3 = get(gca, 'Position');
pos3(2) = pos1(2) - pos3(4) - 0.05;
set(gca, 'Position', pos3);

subplot(2, 2, 4);
xlim([1e2 1e7])
ylim([1e-1 1e1])
set(gca, 'xtick', [1e2 1e3 1e4 1e5 1e6 1e7])
set(gca, 'yaxislocation', 'right')
xlabel('N')
ylabel('T / T(Thrust)');
axis square

pos4 = get(gca, 'Position');
pos4(1) = pos3(1) + pos3(3) - 0.05;
pos4(2) = pos2(2) - pos4(4) - 0.05;
set(gca, 'Position', pos4);

print('-depsc', 'perfmtx.eps');
close all
