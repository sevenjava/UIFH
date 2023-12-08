function tool_saveparfor(path, X)
    % this function is for satisified the save function in parfor
    % path: the path of save file
    % X: the saved data
    save(path, 'X');
end