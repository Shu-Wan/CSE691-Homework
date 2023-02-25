# CSE691 Homework 3, Spiders and flies problem
# Shu Wan

library(tidyverse)
library(magick)

# Default value -----------------------------------------------------------

GRID_WIDTH = 10
GRID_LENGTH = 10
INIT_BOARD = matrix(data = c(7,1,1,7,1,1,9,3,1,3,4,1,5,7,1,2,8,1,9,9,1), byrow = TRUE, ncol = 3)
rownames(INIT_BOARD) <- c(paste0("S", 1:2), paste0("F", 1:5))
colnames(INIT_BOARD) <- c("x", "y", "status")

INIT_STATE <- list(
  stage = 0,
  board = INIT_BOARD,
  current_cost = 0,
  total_cost = 0
)
class(INIT_STATE) <- "state"

MAX_STAGE = 40

# Utility functions -------------------------------------------------------

#' check if `x` is a point (2d vector)
is_point <- function(x){
  length(x) == 2
}

#' Check is all flies are dead
#' @param state a state object
#' @return a logical; `TRUE` if all flies are dead
is_clear <- function(...) {
  UseMethod("is_clear")
}

is_clear.default <- function(board) {
  all(board[3:7, "status"] == 0)
}

is_clear.state <- function(state) {
  stopifnot(class(state) == "state")
  all(state[["board"]][3:7, "status"] == 0)
}


# Move functions -------------------------------------------------------

#' Calculate Manhattan Distance between two points on the grid
#' @param point_a a 2d vector, point a coordinate
#' @param point_b a 2d vector, point b coordinate
#' @return integer, Manhattan distance
distance <- function(point_a, point_b) {
  stopifnot(is_point(point_a), is_point(point_b))
  abs(point_a[1] - point_b[1]) + abs(point_a[2] - point_b[2])
}

#' Given current coordinate, determine permissible movements
#'
#' @description Each agent has 5 movements options available in each stage.
#' However, when agent is at the boarder, some movements is not legal.
#' `move_set` determine all permissible movements given a coordinate.
#'
#' The movement is coded from 1 to 5 in this order: RIGHT, LEFT, UP, DOWN, STAY
#'
#' @param point coordinate
#' @return permissible movements
move_set <- function(point) {
  movement <- 1:5
  if (point[1] == 1) {
    movement <- movement[-which(movement == 2)]
  }
  if (point[1] == GRID_LENGTH) {
    movement <- movement[-which(movement == 1)]
  }
  if (point[2] == 1) {
    movement <- movement[-which(movement == 4)]
  }
  if (point[2] == GRID_WIDTH) {
    movement <- movement[-which(movement == 3)]
  }
  movement
}

move_one <- function(point, movement) {
  stopifnot(length(movement) == 1)
  if (movement == 5) {
    return(point)
  }
  if (movement == 1) {
    return(point + c(1, 0))
  }
  if (movement == 2) {
    return(point + c(-1, 0))
  }
  if (movement == 3) {
    return(point + c(0, 1))
  }
  if (movement == 4) {
    return(point + c(0, -1))
  }
}

#' Move to target point
#' Prefer move horizontally if possible
#'
#' @param current current coordinate
#' @param target target coordinate
#' @return move
move_to <- function(current, target) {
  stopifnot(is_point(current), is_point(target))
  diff <- target - current
  if (diff[1] > 0) {
    return(1)
  } else if (diff[1] < 0 ) {
    return(2)
  } else if (diff[2] > 0) {
    return(3)
  } else if (diff[2] < 0) {
    return(4)
  } else {
    return(5)
  }
}

#' Move `point` with specified `movement`
#' @param points matrix, coordinate
#' @param movements a vector, movement option
#' @return a matrix, each row is an updated point coordinate
move <- function(points, movements) {
  stopifnot(is.matrix(points))
  for (i in seq(nrow(points))) {
    points[i, 1:2] <- move_one(points[i, 1:2], movements[i])
  }
  points
}

# Policy functions --------------------------------------------------------
#' Update flies status given spiders/flies locations
#' @param flies flies matrix
#' @param spiders spider matrix
#' @return new board
update_board <- function(flies, spiders) {
  for (i in seq(nrow(flies))) {
    if (flies[i, "status"] == 0) {
      next
    }
    dist <- sapply(seq(nrow(spiders)), function(x){distance(flies[i, 1:2], spiders[x, 1:2])})
    if (min(dist) == 0) {
      flies[i, "status"] = 0
    }
  }
  rbind(spiders, flies)
}

cost <- function(move, num_stage) {
  sum(move != 5) + num_stage
}

#' given current state and move decisions, generate the new state
#' @param state current state
#' @param move movements
#' @return new state
state_transition <- function(state, move) {
  board <- state[["board"]]
  spiders <- board[1:2, ]
  flies <- board[3:7, ]

  new_spiders <- move(spiders, move)

  state[["board"]] <- update_board(flies, new_spiders)

  state[["stage"]] <- state[["stage"]] + 1
  state[["current_cost"]] <- cost(move, 1)
  state[["total_cost"]] <- state[["total_cost"]] + state[["current_cost"]]
  state
}

# base policy -------------------------------------------------------------

#' @param current
#' @param targets a matrix, each row is a target coordinate
#' @return target location, if tied, pick the first one
nearest_target <- function(current, targets) {
  dis <- sapply(seq(nrow(targets)), function(x){distance(current, targets[x,])})
  targets[which.min(dis), 1:2, drop = FALSE]
}

#' Move spiders based on nearest neighbor heuristics
#' @param spiders spider matrix
#' @param flies flies matrix
nn_move <- function(state) {
  board <- state[["board"]]
  spiders <- board[1:2, ]
  flies <- board[3:7, ]
  flies_alive <- flies[flies[, "status"] == 1, 1:2, drop = FALSE]
  sapply(seq(nrow(spiders)), function(x){
    nn <- nearest_target(spiders[x, 1:2], flies_alive)
    move_to(spiders[x, 1:2], nn)
  })
}

base_policy <- function(state) {
  stopifnot(class(state) == "state")

  if (is_clear(state)) {
    print("GAME OVER")
    return(state)
  }

  # spiders base moves
  base_move <- nn_move(state)
  # update state
  state_transition(state, base_move)
}

# Rollout policy ----------------------------------------------------------

best_move <- function(state, move_set) {
  move_set <- as.matrix(move_set)
  # for each moves, compute the total cost
  cost_to_go <- sapply(seq(nrow(move_set)), function(x){
    new_state <- state_transition(state, move_set[x, ])
    game_to_go <- run_base(new_state)
    # cost-to-go
    game_to_go[[length(game_to_go)]][["total_cost"]]
  })
  move_set[which.min(cost_to_go),]
}

#' find the best move with minimal cost, if ties, pick the moves with highest priority
rollout_move <- function(state) {
  board <- state[["board"]]
  spiders <- board[1:2, ]
  # generate all possible moves
  total_move_set <- lapply(seq(nrow(spiders)), function(x){
    # permissible movements
    move_set(spiders[x, 1:2])
  }) %>% expand.grid()
  best_move(state, total_move_set)
}

rollout_policy <- function(state) {
  stopifnot(class(state) == "state")

  if (is_clear(state)) {
    print("GAME OVER")
    return(state)
  }

  best_move <- rollout_move(state)
  state_transition(state, best_move)
}

# Multiagent policy -------------------------------------------------------
ma_move <- function(state) {
  board <- state[["board"]]
  spiders <- board[1:2, ]
  # base move
  best_move <- nn_move(state)
  # generate all possible moves
  for (i in seq(nrow(spiders))) {
    total_move_set <- expand.grid(move_set(spiders[i, 1:2]), best_move[-i])
    # switch order
    if (i == 2) {
      total_move_set <- total_move_set[, 2:1]
    }
    best_move <- best_move(state, total_move_set)
  }
  best_move
}

ma_policy <- function(state) {
  stopifnot(class(state) == "state")

  if (is_clear(state)) {
    print("GAME OVER")
    return(state)
  }

  ma_move <- ma_move(state)
  state_transition(state, ma_move)
}
# Plot functions ----------------------------------------------------------

# Set default plot theme
theme_set(
  theme_minimal() +
    theme(panel.grid = element_blank(),
          plot.title = element_text(face = "bold"))
)

#' Display current state
#'
#'
display_state <- function(state) {
  stopifnot(class(state) == "state")
  state[["board"]] %>%
    as_tibble(rownames = "agent") %>%
    ggplot(aes(x = x, y = y)) +
    geom_tile(data = expand_grid(x = seq(10), y = seq(10)),
              color = "black", fill = NA, linewidth = .5) +
    geom_text(data = . %>% filter(status == 1),
              aes(label = agent, color = grepl("^S", agent)),
              fontface = 2, show.legend = FALSE) +
    geom_tile(color = "black", fill = NA) +
    scale_color_manual(values = c("#8C1D40", "#FFC627")) +
    scale_y_continuous(breaks = 1:10, trans = "reverse") +
    scale_x_continuous(breaks = 1:10) +
    labs(
      x = "", y = "",
      title = paste("Stage:", state[["stage"]]),
      subtitle = paste("Current Cost:", state[["current_cost"]], "Total Cost:", state[["total_cost"]])
    )
}

#' Create an animated gif of the whole game
#' @param game
#' @param path path to save gif
#' @return
replay <- function(game, path, print = TRUE) {
  stopifnot(class(game) == "game")

  img <- image_graph(600, 340, res = 96)
  lapply(game, function(x){
    print(display_state(x))
  })
  dev.off()
  animation <- image_animate(img, fps = 2, optimize = TRUE)
  if (print) {
    print(animation)
  }
  image_write(animation, path = path, format = "gif")
}

# Experiment functions ---------------------------------------------------

sitrep <- function(game) {
  num_stage <- length(game)
  total_cost <- game[[length(game)]][["total_cost"]]
  steps_per_stage <- (total_cost - num_stage + 1) / (num_stage - 1)
  list(
    "Number of Stage" = num_stage,
    "Total Cost" = total_cost,
    "Average steps per stage" = steps_per_stage
  )
}

run_base <- function(state, max = MAX_STAGE) {
  game <- list(state)
  N <- 1
  while (!(is_clear(state) | N > max)) {
    N <- N + 1
    state <- base_policy(state)
    game <- c(game, list(state))
  }
  class(game) <- "game"
  game
}

run_rollout <- function(state, max = MAX_STAGE) {
  game <- list(state)
  N <- 1
  while (!(is_clear(state) | N > max)) {
    N <- N + 1
    state <- rollout_policy(state)
    game <- c(game, list(state))
  }
  class(game) <- "game"
  game
}

run_ma <- function(state, max = MAX_STAGE) {
  game <- list(state)
  N <- 1
  while (!(is_clear(state) | N > max)) {
    N <- N + 1
    state <- ma_policy(state)
    game <- c(game, list(state))
  }
  class(game) <- "game"
  game
}

# Implementation -------------------------------------------------------------

library(microbenchmark)
microbenchmark(
  run_base(INIT_STATE),
  run_rollout(INIT_STATE),
  run_ma(INIT_STATE)
)

res <- run_base(INIT_STATE)
sitrep(res)
replay(res, "base.gif")

res <- run_rollout(INIT_STATE)
sitrep(res)
replay(res, "rollout.gif")

res <- run_ma(INIT_STATE)
sitrep(res)
replay(res, "ma.gif")
