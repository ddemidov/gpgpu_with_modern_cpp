function relperf()
cpu_data = {'thrust_cpu', 'vexcl_cpu_amd', 'vexcl_cpu_intel', 'viennacl_cpu_amd', 'viennacl_cpu_intel'};
gpu_data = {'thrust_gpu', 'vexcl_1gpu', 'viennacl_gpu'};
tahiti_data = {'thrust_gpu', 'vexcl_1gpu_tahiti', 'viennacl_gpu_tahiti'};

fprintf('--- CPU ----------------------------------------\n');
get_stat(cpu_data)

fprintf('--- Tesla ----------------------------------------\n');
get_stat(gpu_data)

fprintf('--- Tahiti -------------------------------------\n');
get_stat(tahiti_data)

function get_stat(dev_data)

%folder = {'lorenz_ensemble', 'phase_oscillator_chain', 'disordered_ham_lattice'};
folder = {'lorenz_ensemble', 'phase_oscillator_chain'};

min_time = [];
max_time = [];
time = {};
idx = 0;
for lib = dev_data
    idx = idx + 1;
    tm = [];
    for dir = folder
	data = load([cell2mat(dir) '/' cell2mat(lib) '.dat']);
	n = max(unique(data(:,1)));
	I = find(data(:,1) == n);
	avg = median(data(I,2));
	tm = [tm avg];
    end

    fprintf('%s: ', cell2mat(lib));
    fprintf('\t%8.2f', tm);
    fprintf('\n');

    time{idx} = tm;

    if (length(min_time) == 0)
	min_time = tm;
	max_time = tm;
    else
	min_time = min(min_time, tm);
	max_time = max(max_time, tm);
    end
end

fprintf('\n');
for idx = 1:3
    fprintf('%s: ', dev_data{idx});
    fprintf('\t%8.2f', time{idx} ./ min_time);
    fprintf('\n');
end
fprintf('\n');





