% Generate the maze
map = mapMaze(5, 2, 'MapSize', [20 16], 'MapResolution', 5);

% Get map dimensions (in cells)
grid = occupancyMatrix(map); % Convert to matrix (1 = obstacle, 0 = free)
[grid_h, grid_w] = size(grid);

% Extract obstacle coordinates (x,y)
[rows, cols] = find(grid == 1);

obstacles = [cols - 1, rows - 1]; % zero-based coordinates for Python

% robots position
robots = [
    31, 18;
    33, 18;
    33, 22;
    31, 22;
    32, 20
];

% Define other parameters
data.cell_size = 10;
data.grid_w = grid_w;
data.grid_h = grid_h;
data.sense_radius = 8;
data.comm_range = 20;
data.robots = robots;
data.obstacles = obstacles;

% Save as JSON
jsonStr = jsonencode(data);

% Write to file
fid = fopen('maze.json', 'w');
fprintf(fid, '%s', jsonStr);
fclose(fid);

disp('✅ Maze exported to maze.json');