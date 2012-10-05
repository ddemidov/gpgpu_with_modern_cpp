close all
clear all

test = {'thrust_gpu',   'thrust_cpu', ...
	'vexcl_1gpu',   'vexcl_1gpu_tahiti', 'vexcl_cpu', ...
	'viennacl_gpu', 'viennacl_gpu_tahiti', 'viennacl_cpu'};

lgnd = {'Thrust Tesla',   'Thrust CPU', ...
	'VexCL Tesla', 'VexCL Tahiti', 'VexCL CPU', ...
	'ViennaCL Tesla', 'ViennaCL Tahiti', 'ViennaCL CPU'};

style = {'ko-', 'kd-', ...
	 'ko-', 'kv-', 'kd-', ...
	 'ko--', 'kv--', 'kd--'};

fcolor = {'k', 'k', ...
          'w', 'w', 'w', ...
	  'w', 'w', 'w'};

msize = 3;

figure(1)
set(gcf, 'position', [50, 50, 1000, 500]);

subplot(1, 2, 1);
set(gca, 'FontSize', 10);
subplot(1, 2, 2);
set(gca, 'FontSize', 10);

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

    subplot(1, 2, 1);
    loglog(n, avg, style{idx}, 'markersize', msize, 'markerfacecolor', fcolor{idx});
    hold on

    subplot(1, 2, 2);
    loglog(n, avg ./ ref_avg, style{idx}, 'markersize', msize, 'markerfacecolor', fcolor{idx});
    hold on
end

subplot(1, 2, 1);
xlim([1e2 1e7])
set(gca, 'xtick', [1e2 1e3 1e4 1e5 1e6 1e7])
xlabel('N');
ylabel('T (sec)');
h = legend(lgnd, 'location', 'NorthWest');
set(h, 'fontsize', 8);
legend boxoff
axis square

subplot(1, 2, 2);
xlim([1e2 1e7])
set(gca, 'xtick', [1e2 1e3 1e4 1e5 1e6 1e7])
xlabel('N');
ylabel('T / T(Thrust GPU)');
axis square

print('-depsc', 'perfcmp.eps');
