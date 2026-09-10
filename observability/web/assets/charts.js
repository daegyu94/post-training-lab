/* Shared SVG line-chart math for index.html and telemetry.html. Pure functions only — no DOM access. */
const Charts = (() => {
  function linspace(start, end, count) {
    if (count <= 1) return [start];
    const step = (end - start) / (count - 1);
    return Array.from({ length: count }, (_, i) => start + i * step);
  }

  function scaleLinear(value, [domainStart, domainEnd], [rangeStart, rangeEnd]) {
    if (domainEnd === domainStart) return rangeStart;
    return rangeStart + ((value - domainStart) / (domainEnd - domainStart)) * (rangeEnd - rangeStart);
  }

  /* Evenly spaced horizontal gridlines between y0 (first) and y1 (last), as "<path d>" strings. */
  function gridLines(count, { x0, x1, y0, y1 }) {
    return linspace(y0, y1, count).map(y => `M${x0} ${y}H${x1}`);
  }

  /* Points already in pixel space -> an SVG "<path d>" string. */
  function pathFromPoints(points, digits = 1) {
    return points.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(digits)} ${y.toFixed(digits)}`).join(' ');
  }

  /* Points already in pixel space -> an SVG "<polyline points>" string. */
  function polylineFromPoints(points) {
    return points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  }

  return { linspace, scaleLinear, gridLines, pathFromPoints, polylineFromPoints };
})();
if (typeof module !== 'undefined') module.exports = Charts;
