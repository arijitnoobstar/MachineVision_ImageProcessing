% We define a node handle class to represent a label
% A handle class is used so that the parent attribute can be referenced to
% the object itself
classdef LabelNode < handle
   properties
      Parent
      Size
      Label
   end
end