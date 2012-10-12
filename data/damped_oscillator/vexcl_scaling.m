close all
clear all

test = {'vexcl_1gpu', 'vexcl_2gpu', 'vexcl_3gpu', ...
	'vexcl_1gpu_tahiti', 'vexcl_2gpu_tahiti', 'vexcl_3gpu_tahiti'};

lgnd   = {'Tesla  \times 1', 'Tesla  \times 2', 'Tesla  \times 3', ...
          'Tahiti  \times 1', 'Tahiti  \times 2', 'Tahiti  \times 3'};
style  = {'k:', 'ko-', 'ko--', 'k:', 'ko-', 'ko--'};
fcolor = {'w', 'w', 'w', 'k', 'k', 'k'};
msize  = 4;

figure(1)
set(gca, 'FontSize', 26);

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

    if idx == 1 || idx == 4
	ref_avg = avg;
    end

    semilogx(n, ref_avg ./ avg, style{idx}, 'markersize', msize, 'markerfacecolor', fcolor{idx});
    hold on
end

xlim([1e2 1e7])
set(gca, 'xtick', [1e2 1e3 1e4 1e5 1e6 1e7])
xlabel('N');
ylabel('T(1 GPU) / T');
legend(lgnd, 'location', 'NorthWest');
legend boxoff
axis square

print('-depsc', 'scaling.eps');
