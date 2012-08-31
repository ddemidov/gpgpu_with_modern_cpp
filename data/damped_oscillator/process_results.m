close all
clear all

test = {'thrust_cpu', 'thrust_gpu', ...
	'vexcl_cpu', 'vexcl_1gpu', 'vexcl_2gpu', 'vexcl_3gpu'...
	'viennacl_cpu', 'viennacl_gpu' };
style = {'kd-', 'ko-', 'rd-', 'ro-', 'rs-', 'rv-', 'bd-', 'bo-'};
legstr = {};

figure(1)
set(gca, 'FontSize', 18)

idx = 0;
for t = test
    idx = idx + 1;
    data = load([cell2mat(t) '.dat']);
    avg = [];

    n = unique(data(:,1))';
    for i = n
	I = find(data(:,1) == i);
	time = sum(data(I,2)) / length(I);
	avg = [avg time];
    end

    loglog(n, avg, style{idx}, 'linewidth', 1, 'markersize', 6, 'markerfacecolor', 'w');
    hold on

    legstr{idx} = strrep(cell2mat(t), '_', ' ');
end

xlim([1e2 1e7])
set(gca, 'xtick', [1e2 1e3 1e4 1e5 1e6 1e7])
xlabel('N');
ylabel('T (sec)');

legend(legstr, 'location', 'northwest');
legend boxoff

%print('-depsc', 'abs.eps');
