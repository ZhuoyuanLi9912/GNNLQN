function Random_LQN_generator(num_LQNs, output_file, config)
    % Generate a dataset of LQN models and their simulated metrics using LQNS
    %
    % Args:
    %   num_LQNs (int): Number of LQN models to generate
    %   output_file (string): Output file name to save the dataset
    %   config (struct): Configuration for random ranges
    %
    % Returns:
    %   Saves the dataset in a .mat file

    % Default configuration if not provided
    if nargin < 3
        config = struct( ...
            'num_processors', [3, 5], ...  % Range for number of processors
            'tasks_per_processor', [2, 2], ... % Range for tasks per processor
            'entries_per_task', [2, 2], ... % Range for entries per task
            'calls_per_entry', [2, 2]); % Range for entry calls
    end

    % Initialize cell array to store successfully generated LQN models
    LQN_dataset = cell(num_LQNs, 1);

    % Counter for successfully processed LQNs
    successful_count = 0;

    i = 1; % Loop counter for total attempts
    while successful_count < num_LQNs
        try
            % Step 1: Generate a random LQN model
            LQN = generate_random_lqn(config);

            % Step 2: Simulate the LQN using LQNS to calculate metrics
            entry_metrics = simulate_lqn_lqns(LQN);

            % Step 3: Store the metrics in the LQN struct
            LQN.entry_queue_lengths = entry_metrics.queue_lengths;
            LQN.entry_response_times = entry_metrics.response_times;
            LQN.entry_throughputs = entry_metrics.throughputs;

            % Step 4: Save the LQN model with metrics
            successful_count = successful_count + 1; % Increment successful count
            LQN_dataset{successful_count} = LQN; % Store the LQN

            % Display progress
            disp(['Generated and simulated LQN ', num2str(successful_count), ' of ', num2str(num_LQNs)]);

        catch ME
            % Handle the error gracefully and continue to the next iteration
            disp(['Error encountered while processing LQN ', num2str(i), ': ', ME.message]);
            disp('Skipping this LQN and continuing...');
        end

        % Always increment the total attempt counter
        i = i + 1;
    end

    % Save the dataset to a .mat file
    save(output_file, 'LQN_dataset');
    disp(['LQN dataset saved to ', output_file]);
end



function LQN = generate_random_lqn(config)
    % Generate a random LQN model with processors, tasks, entries, and activities
    %
    % Args:
    %   config (struct): Configuration for random ranges
    %
    % Returns:
    %   LQN (struct): A struct representing the LQN model

    % Randomly determine the number of processors
    num_processors = randi(config.num_processors);
    processor_attributes = randi([1, 10], num_processors, 1);  % Multiplicity

    % Initialize tasks and entries
    tasks = [];
    entries = [];
    task_on_processor_edges = [];
    entry_on_task_edges = [];
    entry_call_entry_edges = [];
    entry_call_entry_edge_attributes = [];

    % Step 1: Create Layers, Tasks, and Entries
    layer_entries = cell(num_processors, 1); % Track entries in each processor layer
    for p = 1:num_processors
        % Determine number of tasks for this processor
        num_tasks = randi(config.tasks_per_processor);

        % Store the tasks and entries for this processor
        processor_tasks = [];
        processor_entries = [];

        for t = 1:num_tasks
            % Add task (including multiplicity as the third column)
            think_time = round(rand(1, 2) * 2.9 + 0.1, 1); % Mean think time and SCV: 0.1 to 3.0
            multiplicity = randi([1, 5]); % Random multiplicity between 1 and 5
            tasks = [tasks; think_time, multiplicity];
            processor_tasks = [processor_tasks; size(tasks, 1)];

            % Determine number of entries for this task
            num_entries = randi(config.entries_per_task);

            for e = 1:num_entries
                % Add entry
                service_time = round(rand(1, 2) * 2.9 + 0.1, 1); % Mean service time and SCV: 0.1 to 3.0
                entries = [entries; service_time];
                processor_entries = [processor_entries; size(entries, 1)];

                % Map entry to the task
                entry_on_task_edges = [entry_on_task_edges, [size(entries, 1); size(tasks, 1)]];
            end

            % Map task to the processor
            task_on_processor_edges = [task_on_processor_edges, [size(tasks, 1); p]];
        end

        % Store entries for this layer
        layer_entries{p} = processor_entries;
    end

    % Step 2: Add Calls Between Layers
    for p = 1:(num_processors - 1) % No calls from the last layer
        current_layer_entries = layer_entries{p};
        next_layer_entries = layer_entries{p + 1};

        % Precompute the number of outgoing calls for each entry in the current layer
        current_layer_call_limits = randi(config.calls_per_entry, size(current_layer_entries)); % Use the interval in config
        current_layer_assigned_calls = zeros(size(current_layer_entries)); % Track assigned calls
        source_entry_probabilities = zeros(size(current_layer_entries)); % Track total probabilities per source entry

        % Track assigned edges to avoid duplicates
        existing_edges = containers.Map();

        % Pre-check: Ensure total available call slots can cover next layer entries
        while sum(current_layer_call_limits - current_layer_assigned_calls) < length(next_layer_entries)
            % Increment the call limit for the entry with the fewest remaining slots
            [~, idx] = min(current_layer_call_limits - current_layer_assigned_calls);
            current_layer_call_limits(idx) = current_layer_call_limits(idx) + 1;
        end

        % Ensure every entry in the next layer has at least one incoming call
        for e = 1:length(next_layer_entries)
            target_entry = next_layer_entries(e);

            % Randomly select a source entry with available call slots
            available_indices = find(current_layer_assigned_calls < current_layer_call_limits);
            source_entry_idx = available_indices(randi(length(available_indices)));
            source_entry = current_layer_entries(source_entry_idx);

            % Record the assigned edge
            edge_key = sprintf('%d-%d', source_entry, target_entry);
            existing_edges(edge_key) = true;

            % Generate attributes for this call
            probability = round(rand(1) * 0.9 + 0.1, 1); % Random probability: 0.1 to 1.0
            mean_number_of_calls = round(rand(1) * 2.9 + 0.1, 1); % Random mean: 0.1 to 3.0
            mean_call_time = round(rand(1) * 2.9 + 0.1, 1); % Random mean: 0.1 to 3.0
            scv_call_time = round(rand(1) * 2.9 + 0.1, 1); % Random SCV: 0.1 to 3.0

            % Add the call to the edge list
            entry_call_entry_edges = [entry_call_entry_edges, [source_entry; target_entry]];
            entry_call_entry_edge_attributes = [entry_call_entry_edge_attributes; ...
                probability, mean_number_of_calls, mean_call_time, scv_call_time];

            % Update the assigned call count and probability
            current_layer_assigned_calls(source_entry_idx) = current_layer_assigned_calls(source_entry_idx) + 1;
            source_entry_probabilities(source_entry_idx) = source_entry_probabilities(source_entry_idx) + probability;
        end

        % Add remaining calls for entries in the current layer that have not reached their limit
        for e = 1:length(current_layer_entries)
            source_entry = current_layer_entries(e);

            % Determine how many more calls this entry can make
            remaining_calls = current_layer_call_limits(e) - current_layer_assigned_calls(e);

            if remaining_calls > 0
                % Assign remaining calls
                for c = 1:remaining_calls
                    max_retries = 100; % Set a limit on retries
                    retry_count = 0;

                    % Randomly select a target entry
                    target_entry = next_layer_entries(randi(length(next_layer_entries)));
                    edge_key = sprintf('%d-%d', source_entry, target_entry);

                    % Retry if the edge already exists
                    while isKey(existing_edges, edge_key)
                        retry_count = retry_count + 1;

                        % Break out of the loop if max retries are reached
                        if retry_count > max_retries
                            warning('Exceeded maximum retries when finding unique target entry.');
                            break;
                        end

                        % Retry with a new target entry
                        target_entry = next_layer_entries(randi(length(next_layer_entries)));
                        edge_key = sprintf('%d-%d', source_entry, target_entry);
                    end

                    % Skip this call if retries are exceeded
                    if retry_count > max_retries
                        continue;
                    end

                    % Record the assigned edge
                    existing_edges(edge_key) = true;

                    % Generate attributes for this call
                    probability = round(rand(1) * 0.9 + 0.1, 1); % Random probability: 0.1 to 1.0
                    mean_number_of_calls = round(rand(1) * 2.9 + 0.1, 1); % Random mean: 0.1 to 3.0
                    mean_call_time = round(rand(1) * 2.9 + 0.1, 1); % Random mean: 0.1 to 3.0
                    scv_call_time = round(rand(1) * 2.9 + 0.1, 1); % Random SCV: 0.1 to 3.0

                    % Add the call to the edge list
                    entry_call_entry_edges = [entry_call_entry_edges, [source_entry; target_entry]];
                    entry_call_entry_edge_attributes = [entry_call_entry_edge_attributes; ...
                        probability, mean_number_of_calls, mean_call_time, scv_call_time];

                    % Update the assigned call count and probability
                    current_layer_assigned_calls(e) = current_layer_assigned_calls(e) + 1;
                    source_entry_probabilities(e) = source_entry_probabilities(e) + probability;
                end
            end
        end

        % Normalize probabilities for all source entries
        for i = 1:length(current_layer_entries)
            % Get all edges originating from this source entry
            source_entry = current_layer_entries(i);
            edge_indices = find(entry_call_entry_edges(1, :) == source_entry);

            % Get the total assigned probability for this source entry
            total_probability = sum(entry_call_entry_edge_attributes(edge_indices, 1));

            if total_probability > 0
                % Normalize probabilities for all edges from this source
                normalized_probs = entry_call_entry_edge_attributes(edge_indices, 1) / total_probability;

                % Round probabilities to 1 decimal place
                rounded_probs = round(normalized_probs, 1);

                % Adjust to ensure the sum equals exactly 1.0
                difference = 1.0 - sum(rounded_probs);
                [~, max_idx] = max(rounded_probs); % Adjust the largest probability
                rounded_probs(max_idx) = rounded_probs(max_idx) + difference;

                % Assign the adjusted probabilities back
                entry_call_entry_edge_attributes(edge_indices, 1) = rounded_probs;
            end
        end
    end

    % Create the LQN struct
    LQN = struct();
    LQN.processor_attributes = processor_attributes;
    LQN.task_attributes = tasks;
    LQN.entry_attributes = entries;
    LQN.task_on_processor_edges = task_on_processor_edges;
    LQN.entry_on_task_edges = entry_on_task_edges;
    LQN.entry_call_entry_edges = entry_call_entry_edges;
    LQN.entry_call_entry_edge_attributes = entry_call_entry_edge_attributes;
end




function entry_metrics = simulate_lqn_lqns(LQN)
    % Simulate the LQN and extract metrics for each entry using LQNS
    %
    % Args:
    %   LQN (struct): The LQN model
    %
    % Returns:
    %   entry_metrics (struct): Struct containing queue lengths, response times, and throughputs for each entry

    % Create the LayeredNetwork model
    model = LayeredNetwork('LQN');

    % Step 1: Create processors, tasks, and entries (no calls yet)
    processors = cell(size(LQN.processor_attributes, 1), 1);
    tasks = cell(size(LQN.task_attributes, 1), 1);
    entries = cell(size(LQN.entry_attributes, 1), 1);
    activities = cell(size(LQN.entry_attributes, 1), 1); % Store primary activities

    % Create processors
    for i = 1:size(LQN.processor_attributes, 1)
        processors{i} = Processor(model, ['P', num2str(i)], LQN.processor_attributes(i), SchedStrategy.PS);
    end

    % Create tasks
    for i = 1:size(LQN.task_attributes, 1)
        multiplicity = LQN.task_attributes(i, 3); % Extract multiplicity
        if LQN.task_on_processor_edges(2, i) == 1
            sched_strategy = SchedStrategy.REF; % First processor
        else
            sched_strategy = SchedStrategy.FCFS; % Other processors
        end

        tasks{i} = Task(model, ['T', num2str(i)], multiplicity, sched_strategy).on(processors{LQN.task_on_processor_edges(2, i)});
        tasks{i}.setThinkTime(APH.fitMeanAndSCV(LQN.task_attributes(i, 1), LQN.task_attributes(i, 2)));
    end

    % Create entries and their primary activities
    for i = 1:size(LQN.entry_attributes, 1)
        task_id = LQN.entry_on_task_edges(2, i); % Get task ID for this entry
        entries{i} = Entry(model, ['E', num2str(i)]).on(tasks{task_id});

        % Create primary activity for this entry
        activities{i} = Activity(model, ['A', num2str(i)], APH.fitMeanAndSCV(LQN.entry_attributes(i, 1), LQN.entry_attributes(i, 2))) ...
            .on(tasks{task_id}).boundTo(entries{i});
    end

    % Step 2: Add calls between entries
    call_activity = cell(size(LQN.entry_attributes, 1), 1); % Store call activities for each entry
    global_call_counter = 1; % Initialize a global call counter

    for i = 1:size(LQN.entry_attributes, 1)
        % Check if this entry makes any calls
        outgoing_call_indices = find(LQN.entry_call_entry_edges(1, :) == i); % Find all calls originating from this entry
        task_id = LQN.entry_on_task_edges(2, i); % Get task ID for this entry
        is_top_layer_task = (LQN.task_on_processor_edges(2, task_id) == 1); % Check if this task is on the top layer

        if isempty(outgoing_call_indices) % Bottom layer: no calls
            activities{i}.repliesTo(entries{i});
        else
            % This entry makes calls: create call activities and an OrFork
            target_activities = {};
            probabilities = [];
            call_activity{i} = cell(length(outgoing_call_indices), 1); % Initialize call activities for this entry

            for j = 1:length(outgoing_call_indices)
                call_index = outgoing_call_indices(j);
                target_entry = LQN.entry_call_entry_edges(2, call_index); % Target entry index
                probability = LQN.entry_call_entry_edge_attributes(call_index, 1); % Call probability
                mean_number_of_calls = LQN.entry_call_entry_edge_attributes(call_index, 2); % Mean number of calls
                mean_call_time = LQN.entry_call_entry_edge_attributes(call_index, 3); % Mean call time
                scv_call_time = LQN.entry_call_entry_edge_attributes(call_index, 4); % SCV of call time

                % Create a call activity with a unique name and appropriate attributes
                call_name = ['Call', num2str(global_call_counter)];
                if is_top_layer_task
                    % Do not include repliesTo for top-layer tasks
                    call_activity{i}{j} = Activity(model, call_name, APH.fitMeanAndSCV(mean_call_time, scv_call_time)) ...
                        .on(tasks{task_id}).synchCall(entries{target_entry}, mean_number_of_calls);
                else
                    % Include repliesTo for other tasks
                    call_activity{i}{j} = Activity(model, call_name, APH.fitMeanAndSCV(mean_call_time, scv_call_time)) ...
                        .on(tasks{task_id}).synchCall(entries{target_entry}, mean_number_of_calls).repliesTo(entries{i});
                end

                % Increment the global call counter
                global_call_counter = global_call_counter + 1;

                % Store the call activity and its probability
                target_activities{end + 1} = call_activity{i}{j}; % Add to target activities
                probabilities(end + 1) = probability; % Add to probabilities
            end

            % Add OrFork precedence for this entry using the provided probabilities
            tasks{task_id}.addPrecedence(ActivityPrecedence.OrFork(activities{i}, target_activities, probabilities));
        end
    end

    % Solve the model using LQNS
    options = SolverLQNS.defaultOptions;
    options.method = 'lqns';
    solver = SolverLQNS(model, options);


    % Extract metrics for entries
    avg_table = solver.getAvgTable();

    % Calculate row range for entries
    num_processors = size(LQN.processor_attributes, 1);
    num_tasks = size(LQN.task_attributes, 1);
    num_entries = size(LQN.entry_attributes, 1);

    entry_start_row = num_processors + num_tasks + 1;
    entry_end_row = entry_start_row + num_entries - 1;

    % Extract relevant rows for entries
    entry_rows = avg_table(entry_start_row:entry_end_row, :);

    % Extract queue lengths, response times, and throughputs
    queue_lengths = table2array(entry_rows(:, 3));   % 3rd column: queue length
    response_times = table2array(entry_rows(:, 5)); % 5th column: response time
    throughputs = table2array(entry_rows(:, 7));    % 7th column: throughput

    % Store metrics in a struct
    entry_metrics = struct();
    entry_metrics.queue_lengths = queue_lengths;
    entry_metrics.response_times = response_times;
    entry_metrics.throughputs = throughputs;
end

