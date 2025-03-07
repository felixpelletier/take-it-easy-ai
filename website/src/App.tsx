import { useEffect, useMemo, useRef, useState } from "react";
import { LoadingOverlay, Box, Container, Modal, Grid } from "@mantine/core";
import "./App.css";

import * as ort from "onnxruntime-web";
import { getBoardScore } from "./Game";
ort.env.wasm.wasmPaths = "./";

function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [session, setSession] = useState<ort.InferenceSession>();

  const tileset = useMemo(() => createTileset(), []);
  const board = useRef(new Array(19).fill(null) as (number | null)[]);
  const [boardView, setBoardView] = useState(
    new Array(19).fill(null) as ([number, number, number] | null)[],
  );
  const [score, setScore] = useState(0);

  const [nextTileIndex, setNextTileIndex] = useState<number | null>(null);
  const placedPosisionsIndex = useRef<number[]>([]);
  const isAlreadyPlaying = useRef(false);

  const [resetConfirmation, setResetConfirmation] = useState(false);
  const resetConfirmationTimeout = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );

  const [tileSelectorOpened, setTileSelectorOpened] = useState(false);

  useEffect(() => {
    (async () => {
      setSession(
        await ort.InferenceSession.create(
          "takeiteasy_1740792046_1500000.q8.onnx",
        ),
      );
      setIsLoading(false);
      resetGame();
    })();
  }, []);

  function randomInt(min: number, max: number) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  function resetGame() {
    board.current.fill(null) as (number | null)[];
    placedPosisionsIndex.current = [];
    setNextTileIndex(randomInt(0, tileset.length - 1));
    refreshBoardView();
  }

  function pressResetButton() {
    if (resetConfirmation) {
      resetGame();
      setResetConfirmation(false);
      if (resetConfirmationTimeout.current != null) {
        clearTimeout(resetConfirmationTimeout.current);
        resetConfirmationTimeout.current = null;
      }
    } else {
      resetConfirmationTimeout.current = setTimeout(() => {
        setResetConfirmation(false);
      }, 2000);
      setResetConfirmation(true);
    }
  }

  async function playOnce() {
    if (session == null) return;
    if (nextTileIndex == null) return;
    if (isAlreadyPlaying.current) return;
    if (isLoading) return;
    isAlreadyPlaying.current = true;

    try {
      const state = [
        ...board.current.flatMap((b) => (b ? tileset[b] : [0, 0, 0])),
        ...tileset[nextTileIndex],
      ];

      const tensor = new ort.Tensor("float32", Float32Array.from(state), [
        1,
        3 * 20,
      ]);

      const results = await session.run({ input: tensor });

      const resultsData = Array.from(results.linear_7.data as Float32Array);

      console.log(board.current);

      const actionIndex = resultsData
        .map((val, i) => [val, i])
        .sort((a, b) => b[0] - a[0])
        .map((valueAndIndex) => valueAndIndex[1])
        .filter((actionIndex) => board.current[actionIndex] == null)[0];

      console.log(actionIndex);
      console.log(nextTileIndex);

      board.current[actionIndex] = nextTileIndex;
      placedPosisionsIndex.current.push(actionIndex);

      if (board.current.includes(null)) {
        const validTiles = [...Array(tileset.length).keys()].filter(
          (tileIndex) => !board.current.includes(tileIndex),
        );

        setNextTileIndex(validTiles[randomInt(0, validTiles.length - 1)]);
      } else {
        setNextTileIndex(null);
      }
      refreshBoardView();
    } finally {
      isAlreadyPlaying.current = false;
    }
  }

  function refreshBoardView() {
    setBoardView(board.current.map((b) => (b != null ? tileset[b] : null)));
    setScore(getBoardScore(board.current, tileset));
  }

  function undo() {
    if (placedPosisionsIndex.current.length === 0) return;
    const lastPlacedIndex =
      placedPosisionsIndex.current[placedPosisionsIndex.current.length - 1];

    placedPosisionsIndex.current.pop();
    setNextTileIndex(board.current[lastPlacedIndex]);
    board.current[lastPlacedIndex] = null;
    refreshBoardView();
  }

  return (
    <>
      <Modal
        opened={tileSelectorOpened}
        onClose={() => setTileSelectorOpened(false)}
        title="Tile selector"
        size={"xs"}
      >
        <Grid columns={3}>
          {tileset.map((tile, tileIndex) => {
            const isEnabled = !board.current.includes(tileIndex);
            return (
              <Grid.Col span={1}>
                <Box
                  onClick={() => {
                    if (!isEnabled) return;
                    setNextTileIndex(tileIndex);
                    setTileSelectorOpened(false);
                  }}
                >
                  <Tile tile={isEnabled ? tile : null} />
                </Box>
              </Grid.Col>
            );
          })}
        </Grid>
      </Modal>

      <Box pos="relative">
        <LoadingOverlay
          visible={isLoading}
          zIndex={1000}
          overlayProps={{ radius: "sm", blur: 2 }}
        />
        <Container className="app">
          <div
            style={{
              maxHeight: "70%",
              padding: "1rem",
              position: "static",
            }}
          >
            <div className="board">
              <BoardTile row_step={0} column_step={0} tile={boardView[0]} />

              <BoardTile row_step={1} column_step={-1} tile={boardView[1]} />
              <BoardTile row_step={1} column_step={1} tile={boardView[2]} />

              <BoardTile row_step={2} column_step={-2} tile={boardView[3]} />
              <BoardTile row_step={2} column_step={0} tile={boardView[4]} />
              <BoardTile row_step={2} column_step={2} tile={boardView[5]} />

              <BoardTile row_step={3} column_step={-1} tile={boardView[6]} />
              <BoardTile row_step={3} column_step={1} tile={boardView[7]} />

              <BoardTile row_step={4} column_step={-2} tile={boardView[8]} />
              <BoardTile row_step={4} column_step={0} tile={boardView[9]} />
              <BoardTile row_step={4} column_step={2} tile={boardView[10]} />

              <BoardTile row_step={5} column_step={-1} tile={boardView[11]} />
              <BoardTile row_step={5} column_step={1} tile={boardView[12]} />

              <BoardTile row_step={6} column_step={-2} tile={boardView[13]} />
              <BoardTile row_step={6} column_step={0} tile={boardView[14]} />
              <BoardTile row_step={6} column_step={2} tile={boardView[15]} />

              <BoardTile row_step={7} column_step={-1} tile={boardView[16]} />
              <BoardTile row_step={7} column_step={1} tile={boardView[17]} />

              <BoardTile row_step={8} column_step={0} tile={boardView[18]} />
            </div>
          </div>
          <div
            style={{
              padding: "1em",
              display: "flex",
              flexDirection: "row",
              justifyContent: "center",
            }}
          >
            <div
              style={{
                flexGrow: 1,
                display: "flex",
                flexDirection: "column",
                justifyContent: "stretch",
                alignItems: "stretch",
              }}
            >
              <p className="" style={{}}>
                Score: {score}
              </p>
              <div
                className="tile"
                style={{
                  position: "relative",
                  maxHeight: "10em",
                  margin: "0 auto",
                }}
              >
                <Box
                  onClick={() => {
                    if (nextTileIndex == null) return; // Means the game is over
                    setTileSelectorOpened(true);
                  }}
                >
                  <Tile
                    tile={nextTileIndex != null ? tileset[nextTileIndex] : null}
                  />
                </Box>
              </div>
            </div>
            <button
              className="button playButton"
              style={{ flex: 6 }}
              onClick={() => playOnce()}
            >
              Play
            </button>
            <button
              className="button playButton"
              style={{ flex: 1 }}
              onClick={() => undo()}
            >
              Undo
            </button>
            <button
              className="button resetButton"
              style={{ flex: 1 }}
              onClick={() => pressResetButton()}
            >
              {resetConfirmation ? "Sure?" : "Reset"}
            </button>
          </div>
        </Container>
      </Box>
    </>
  );
}

function BoardTile(props: {
  row_step: number;
  column_step: number;
  tile: [number, number, number] | null;
}) {
  return (
    <>
      <Container
        className="boardTile"
        style={
          {
            "--xstep": props.column_step,
            "--ystep": props.row_step,
          } as React.CSSProperties
        }
      >
        <Tile tile={props.tile} />
      </Container>
    </>
  );
}

function Tile(props: { tile: [number, number, number] | null }) {
  function Values() {
    if (props.tile == null) return null;

    return (
      <>
        <span
          className="tileNumber"
          style={{
            left: "calc(50% - 0.25em)",
            top: "calc(50% - 40% * sin(90deg) - 0.25em",
          }}
        >
          {props.tile[0]}
        </span>
        <span
          className="tileNumber"
          style={{
            left: "calc(50% + 30% * cos(150deg) - 0.25em)",
            top: "calc(50% + 30% * sin(150deg) - 0.25em",
          }}
        >
          {props.tile[1]}
        </span>
        <span
          className="tileNumber"
          style={{
            left: "calc(50% + 30% * cos(30deg) - 0.25em)",
            top: "calc(50% + 30% * sin(30deg) - 0.25em",
          }}
        >
          {props.tile[2]}
        </span>
      </>
    );
  }

  const radius = 1;
  const points = [0, 1, 2, 3, 4, 5, 6].map((_, i) => {
    var angle_deg = 60 * i;
    var angle_rad = (Math.PI / 180) * angle_deg;
    return [radius * Math.cos(angle_rad), radius * Math.sin(angle_rad)];
  });

  const pointsText = points.map((p) => p.join(",")).join(" ");

  const width = 2 * radius;
  const height = 2 * Math.cos(Math.PI / 6);

  const viewBox = [-(width / 2), -(height / 2), width, height].join(" ");

  return (
    <>
      <Box component="span" pos={"relative"} ta={"left"} display={"block"}>
        <svg
          width="100%"
          height="100%"
          viewBox={viewBox}
          className={`tile ${props.tile ? "filled" : "empty"}`}
          style={{
            stroke: "red",
            strokeWidth: "0.1rem",
          }}
        >
          <polygon vectorEffect={"non-scaling-stroke"} points={pointsText} />
        </svg>
        <Values />
      </Box>
    </>
  );
}

function createTileset(): [number, number, number][] {
  const result: [number, number, number][] = [];
  for (const i of [1, 5, 9]) {
    for (const j of [2, 6, 7]) {
      for (const k of [3, 4, 8]) {
        result.push([i, j, k]);
      }
    }
  }
  return result;
}

export default App;
