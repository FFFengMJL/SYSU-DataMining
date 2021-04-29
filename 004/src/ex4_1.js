let data = [
  { x: 5.9, y: 3.2 },
  { x: 4.6, y: 2.9 },
  { x: 6.2, y: 2.8 },
  { x: 4.7, y: 3.2 },
  { x: 5.5, y: 4.2 },
  { x: 5.0, y: 3.0 },
  { x: 4.9, y: 3.1 },
  { x: 6.7, y: 3.1 },
  { x: 5.1, y: 3.8 },
  { x: 6.0, y: 3.0 },
];

let clusters = [
  { color: "red", x: 6.2, y: 3.2 },
  { color: "green", x: 6.6, y: 3.7 },
  { color: "blue", x: 6.5, y: 3.0 },
];

function distance(point, center) {
  return Math.sqrt(
    Math.pow(point.x - center.x, 2) + Math.pow(point.y - center.y, 2)
  );
}

function updateClusters() {
  let tmp = {
    red: [],
    blue: [],
    green: [],
  };
  for (const point of data) {
    let redDistance = distance(point, clusters[0]);
    let greenDistance = distance(point, clusters[1]);
    let blueDistance = distance(point, clusters[2]);

    if (redDistance < greenDistance && redDistance < blueDistance) {
      tmp.red.push(point);
    } else if (greenDistance < redDistance && greenDistance < blueDistance) {
      tmp.green.push(point);
    } else {
      tmp.blue.push(point);
    }
  }

  console.log(`重新划分后`, tmp);

  for (const cluster of clusters) {
    let newCenter = { x: 0, y: 0 };
    for (const point of tmp[cluster.color]) {
      newCenter.x += point.x;
      newCenter.y += point.y;
    }

    cluster.x = newCenter.x / tmp[cluster.color].length;
    cluster.y = newCenter.y / tmp[cluster.color].length;
  }
}

function iter(times) {
  let n = times;
  console.log(`初始数据为：`, clusters);
  while (n--) {
    console.log(`\n第${times - n}次迭代`);
    updateClusters();
    console.log(clusters);
  }
}

iter(5);
