import java.util.*;

public class PuzzleSolver {

    private static int nodesCreated;
    private static String allow;
    private static int boards;

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("\nDeseja mostrar todos os nodos sendo criados?\n\n- sim\n- nao\n");
        allow = sc.nextLine();
        System.out.println("\n\n");
        int[][][] initialMatrices = {
                { { 1, 2, 3 }, { 4, 5, 6 }, { 0, 7, 8 } },
                { { 1, 3, 0 }, { 4, 2, 5 }, { 7, 8, 6 } },
                { { 1, 3, 5 }, { 2, 6, 0 }, { 4, 7, 8 } },
                { { 1, 8, 3 }, { 4, 2, 6 }, { 7, 5, 0 } },
                { { 1, 2, 3 }, { 7, 0, 6 }, { 4, 8, 5 } }
        };
        int[][] finalMatrix = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 0 } };
        List<String> results = new ArrayList<>();
        boards = initialMatrices.length;
        for (int[][] initialMatrix : initialMatrices) {

            int x = -1, y = -1;
            for (int i = 0; i < initialMatrix.length; i++) {
                for (int j = 0; j < initialMatrix[i].length; j++) {
                    if (initialMatrix[i][j] == 0) {
                        x = i;
                        y = j;
                        break;
                    }
                }
                if (x != -1)
                    break;
            }

            // Breadth-first Search
            long startTime = System.currentTimeMillis();
            solveBreadthFirst(initialMatrix, finalMatrix, x, y);
            long endTime = System.currentTimeMillis();
            results.add(printNodesCreated("Breadth-first Search", endTime - startTime));

            nodesCreated = 0; // Reset counter for the next algorithm

            // Depth-first Search
            startTime = System.currentTimeMillis();
            solveDepthFirst(initialMatrix, finalMatrix, x, y);
            endTime = System.currentTimeMillis();
            results.add(printNodesCreated("Depth-first Search", endTime - startTime));

            nodesCreated = 0;

            // Greedy best-first Search
            startTime = System.currentTimeMillis();
            solveGreedyBestFirst(initialMatrix, finalMatrix, x, y);
            endTime = System.currentTimeMillis();
            results.add(printNodesCreated("Greedy best-first Search", endTime - startTime));

            nodesCreated = 0;

            // A* Search
            startTime = System.currentTimeMillis();
            solveAStar(initialMatrix, finalMatrix, x, y);
            endTime = System.currentTimeMillis();
            results.add(printNodesCreated("A* Search", endTime - startTime) + "\n");
        }
        int resultIndex = 0;
        for (int i = 1; i <= boards; i++) {
            System.out.println("\nTabuleiro " + i);
            for (int j = 0; j < 4; j++) {
                if (resultIndex < results.size()) {
                    System.out.println(results.get(resultIndex++));
                }
            }
            System.out.println();
        }

    }

    public static void solveBreadthFirst(int[][] initialMatrix, int[][] finalMatrix, int x, int y) {
        Queue<Node> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();

        Node initialNode = new Node(initialMatrix, x, y, null, 0);
        queue.add(initialNode);
        visited.add(Arrays.deepToString(initialMatrix));

        while (!queue.isEmpty()) {
            Node currentNode = queue.poll();

            if (Arrays.deepEquals(currentNode.matrix, finalMatrix)) {
                printSolution(currentNode, "\nBreadth First:\n");
                return;
            }

            List<Node> neighbors = getNeighbors(currentNode, finalMatrix.length, finalMatrix[0].length);
            for (Node neighbor : neighbors) {
                String neighborKey = Arrays.deepToString(neighbor.matrix);
                if (!visited.contains(neighborKey)) {
                    queue.add(neighbor);
                    visited.add(neighborKey);
                    nodesCreated++;
                }
            }
        }

        System.out.println("Solution not found");
    }

    public static void solveDepthFirst(int[][] initialMatrix, int[][] finalMatrix, int x, int y) {
        Set<String> visited = new HashSet<>();
        Stack<Node> stack = new Stack<>();

        Node initialNode = new Node(initialMatrix, x, y, null, 0);
        stack.push(initialNode);

        while (!stack.isEmpty()) {
            Node currentNode = stack.pop();

            if (Arrays.deepEquals(currentNode.matrix, finalMatrix)) {
                printSolution(currentNode, "\nDepth First:\n");
                return;
            }

            String currentKey = Arrays.deepToString(currentNode.matrix);
            if (!visited.contains(currentKey)) {
                visited.add(currentKey);
                nodesCreated++;

                List<Node> neighbors = getNeighbors(currentNode, finalMatrix.length, finalMatrix[0].length);
                stack.addAll(neighbors);
            }
        }

        System.out.println("Solution not found");
    }

    public static void solveGreedyBestFirst(int[][] initialMatrix, int[][] finalMatrix, int x, int y) {
        PriorityQueue<Node> priorityQueue = new PriorityQueue<>(
                Comparator.comparingInt(node -> heuristic(node.matrix, finalMatrix)));
        Set<String> visited = new HashSet<>();

        Node initialNode = new Node(initialMatrix, x, y, null, 0);
        priorityQueue.add(initialNode);
        visited.add(Arrays.deepToString(initialMatrix));

        while (!priorityQueue.isEmpty()) {
            Node currentNode = priorityQueue.poll();

            if (Arrays.deepEquals(currentNode.matrix, finalMatrix)) {
                printSolution(currentNode, "\nGreedy Best First:\n");
                return;
            }

            List<Node> neighbors = getNeighbors(currentNode, finalMatrix.length, finalMatrix[0].length);
            for (Node neighbor : neighbors) {
                String neighborKey = Arrays.deepToString(neighbor.matrix);
                if (!visited.contains(neighborKey)) {
                    priorityQueue.add(neighbor);
                    visited.add(neighborKey);
                    nodesCreated++;
                }
            }
        }

        System.out.println("Solution not found");
    }

    public static void solveAStar(int[][] initialMatrix, int[][] finalMatrix, int x, int y) {
        PriorityQueue<Node> priorityQueue = new PriorityQueue<>(
                Comparator.comparingInt(node -> node.cost + heuristic(node.matrix, finalMatrix)));
        Set<String> visited = new HashSet<>();

        Node initialNode = new Node(initialMatrix, x, y, null, 0);
        priorityQueue.add(initialNode);
        visited.add(Arrays.deepToString(initialMatrix));

        while (!priorityQueue.isEmpty()) {
            Node currentNode = priorityQueue.poll();

            if (Arrays.deepEquals(currentNode.matrix, finalMatrix)) {
                printSolution(currentNode, "\nA Star:\n");
                return;
            }

            List<Node> neighbors = getNeighbors(currentNode, finalMatrix.length, finalMatrix[0].length);
            for (Node neighbor : neighbors) {
                String neighborKey = Arrays.deepToString(neighbor.matrix);
                if (!visited.contains(neighborKey)) {
                    priorityQueue.add(neighbor);
                    visited.add(neighborKey);
                    nodesCreated++;
                }
            }
        }

        System.out.println("Solution not found");
    }

    private static int heuristic(int[][] matrix, int[][] finalMatrix) {
        int h = 0;
        int rows = matrix.length;
        int cols = matrix[0].length;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int value = matrix[i][j];
                if (value != 0) {
                    int goalRow = (value - 1) / cols;
                    int goalCol = (value - 1) % cols;
                    h += Math.abs(i - goalRow) + Math.abs(j - goalCol);
                }
            }
        }

        return h;
    }

    private static void printSolution(Node node, String function) {
        if (allow.equals("sim")) {
            System.out.println(function);
            Stack<Node> stack = new Stack<>();
            while (node != null) {
                stack.push(node);
                node = node.parent;
            }

            while (!stack.isEmpty()) {
                Node current = stack.pop();
                printMatrix(current.matrix);
                System.out.println();
            }
        }
    }

    private static void printMatrix(int[][] matrix) {
        for (int[] row : matrix) {
            for (int num : row) {
                System.out.print(num + " ");
            }
            System.out.println();
        }
    }

    private static List<Node> getNeighbors(Node node, int rows, int cols) {
        List<Node> neighbors = new ArrayList<>();

        int[][] directions = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
        for (int[] dir : directions) {
            int newX = node.x + dir[0];
            int newY = node.y + dir[1];

            if (newX >= 0 && newX < rows && newY >= 0 && newY < cols) {
                int[][] newMatrix = copyMatrix(node.matrix);
                newMatrix[node.x][node.y] = node.matrix[newX][newY];
                newMatrix[newX][newY] = 0;

                neighbors.add(new Node(newMatrix, newX, newY, node, node.cost + 1));
            }
        }

        return neighbors;
    }

    private static int[][] copyMatrix(int[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        int[][] copy = new int[rows][cols];

        for (int i = 0; i < rows; i++) {
            System.arraycopy(matrix[i], 0, copy[i], 0, cols);
        }

        return copy;
    }

    private static String printNodesCreated(String algorithm, long timeElapsed) {
        return (algorithm + " - Nodes Created: " + nodesCreated + ", Time Elapsed: " + timeElapsed + "ms");
    }

    static class Node {
        int[][] matrix;
        int x, y;
        Node parent;
        int cost;

        Node(int[][] matrix, int x, int y, Node parent, int cost) {
            this.matrix = matrix;
            this.x = x;
            this.y = y;
            this.parent = parent;
            this.cost = cost;
        }
    }
}