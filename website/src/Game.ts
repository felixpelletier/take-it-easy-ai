type BoardCell = [number, number, number] | null;

export function getBoardScore(rawBoard: (number | null)[], tileset: BoardCell[]): number {
  const board = rawBoard.map((b) => (b != null ? tileset[b] : null))
  let score = 0;

  if (
    board[3] !== null &&
    board[8] !== null &&
    board[13] !== null &&
    board[3][0] === board[8][0] &&
    board[8][0] === board[13][0]
  ) {
    score += 3 * board[3][0];
  }

  if (
    board[5] !== null &&
    board[10] !== null &&
    board[15] !== null &&
    board[5][0] === board[10][0] &&
    board[10][0] === board[15][0]
  ) {
    score += 3 * board[5][0];
  }

  if (
    board[1] !== null &&
    board[6] !== null &&
    board[11] !== null &&
    board[16] !== null &&
    board[1][0] === board[6][0] &&
    board[1][0] === board[11][0] &&
    board[1][0] === board[16][0]
  ) {
    score += 4 * board[1][0];
  }

  if (
    board[2] !== null &&
    board[7] !== null &&
    board[12] !== null &&
    board[17] !== null &&
    board[2][0] === board[7][0] &&
    board[2][0] === board[12][0] &&
    board[2][0] === board[17][0]
  ) {
    score += 4 * board[2][0];
  }

  if (
    board[0] !== null &&
    board[4] !== null &&
    board[9] !== null &&
    board[14] !== null &&
    board[18] !== null &&
    board[0][0] === board[4][0] &&
    board[0][0] === board[9][0] &&
    board[0][0] === board[14][0] &&
    board[0][0] === board[18][0]
  ) {
    score += 5 * board[0][0];
  }

  if (
    board[0] !== null &&
    board[1] !== null &&
    board[3] !== null &&
    board[0][1] === board[1][1] &&
    board[0][1] === board[3][1]
  ) {
    score += 3 * board[0][1];
  }

  if (
    board[15] !== null &&
    board[17] !== null &&
    board[18] !== null &&
    board[15][1] === board[17][1] &&
    board[15][1] === board[18][1]
  ) {
    score += 3 * board[15][1];
  }

  if (
    board[10] !== null &&
    board[12] !== null &&
    board[14] !== null &&
    board[16] !== null &&
    board[10][1] === board[12][1] &&
    board[10][1] === board[14][1] &&
    board[10][1] === board[16][1]
  ) {
    score += 4 * board[10][1];
  }

  if (
    board[2] !== null &&
    board[4] !== null &&
    board[6] !== null &&
    board[8] !== null &&
    board[2][1] === board[4][1] &&
    board[2][1] === board[6][1] &&
    board[2][1] === board[8][1]
  ) {
    score += 4 * board[2][1];
  }

  if (
    board[5] !== null &&
    board[7] !== null &&
    board[9] !== null &&
    board[11] !== null &&
    board[13] !== null &&
    board[5][1] === board[7][1] &&
    board[5][1] === board[9][1] &&
    board[5][1] === board[11][1] &&
    board[5][1] === board[13][1]
  ) {
    score += 5 * board[5][1];
  }

  if (
    board[0] !== null &&
    board[2] !== null &&
    board[5] !== null &&
    board[0][2] === board[2][2] &&
    board[0][2] === board[5][2]
  ) {
    score += 3 * board[0][2];
  }

  if (
    board[13] !== null &&
    board[16] !== null &&
    board[18] !== null &&
    board[13][2] === board[16][2] &&
    board[13][2] === board[18][2]
  ) {
    score += 3 * board[13][2];
  }

  if (
    board[1] !== null &&
    board[4] !== null &&
    board[7] !== null &&
    board[10] !== null &&
    board[1][2] === board[4][2] &&
    board[1][2] === board[7][2] &&
    board[1][2] === board[10][2]
  ) {
    score += 4 * board[1][2];
  }

  if (
    board[8] !== null &&
    board[11] !== null &&
    board[14] !== null &&
    board[17] !== null &&
    board[8][2] === board[11][2] &&
    board[8][2] === board[14][2] &&
    board[8][2] === board[17][2]
  ) {
    score += 4 * board[8][2];
  }

  if (
    board[3] !== null &&
    board[6] !== null &&
    board[9] !== null &&
    board[12] !== null &&
    board[15] !== null &&
    board[3][2] === board[6][2] &&
    board[3][2] === board[9][2] &&
    board[3][2] === board[12][2] &&
    board[3][2] === board[15][2]
  ) {
    score += 5 * board[3][2];
  }

  return score;
}
