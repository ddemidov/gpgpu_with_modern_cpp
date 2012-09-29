close all
clear all

test   = {'vexcl_1gpu_tahiti', 'vexcl_2gpu_tahiti', 'vexcl_3gpu_tahiti'};
lgnd   = {'VexCL 1 GPU', 'VexCL 2 GPUs', 'VexCL 3 GPUs'};
style  = {'ko:', 'ko-', 'ko--'};
fcolor = {'w', 'w', 'w'};
msize  = 3;

figure(1)
set(gca, 'FontSize', 18);

idx = 0;
for t = test
    idx = idx + 1;
    data = load([cell2mat(t) '.dat']);
    avg = [];

    n = unique(data(:,1))';
    for i = n
	I = find(data(:,1) == i);
	time = median(data(I,2));
	avg = [avg time];
    end

    if idx == 1
	ref_avg = avg;
    end

    semilogx(n, ref_avg ./ avg, style{idx}, 'markersize', msize, 'markerfacecolor', fcolor{idx});
    hold on
end

xlim([1e2 1e7])
ylim([0 1.1])
set(gca, 'xtick', [1e2 1e3 1e4 1e5 1e6 1e7])
xlabel('N');
ylabel('T(1 GPU) / T');
h = legend(lgnd, 'location', 'SouthEast');
set(h, 'FontSize', 18);
legend boxoff
axis square

print('-depsc', 'tahiti_scaling.eps');
