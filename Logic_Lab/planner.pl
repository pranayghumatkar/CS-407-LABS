% CS-407 AI Laboratory - Logical Planning
% Optional extension (Tasks 6-8): Prolog as an independent logical verifier.
%
% Student : Pranay Ghumatkar
% Roll No : 2024A3PS0328G
%
% Load with:  swipl planner.pl
% Query with: ?- can_move(a,b).

% ---------------------------------------------------------------------------
% Task 6 - warehouse connectivity (facts)
% ---------------------------------------------------------------------------
connected(a, b).
connected(b, a).
connected(b, c).
connected(c, b).

% Rule: adjacency implies a legal move.   Connected(X,Y) => CanMove(X,Y)
can_move(X, Y) :- connected(X, Y).

% ---------------------------------------------------------------------------
% Task 7 - check a move proposed by the Python planner
% ---------------------------------------------------------------------------
valid_move(X, Y) :- connected(X, Y).

%   ?- valid_move(a,b).   -> true
%   ?- valid_move(b,c).   -> true
%   ?- valid_move(a,c).   -> false   (no connected(a,c) fact, no path rule)
%   ?- valid_move(a,c).   in a plan Move(a,c) is therefore NOT supported.

% ---------------------------------------------------------------------------
% Task 8 - a chain of implications
% ---------------------------------------------------------------------------
wet_road.
slippery :- wet_road.
reduce_speed :- slippery.

%   ?- reduce_speed.  -> true
%   Fact => Rule => Rule => Conclusion:
%     WetRoad => Slippery => ReduceSpeed
