/* Deterministic, bounded synthetic samples; fixture configuration stays fixed. */
const DemoTelemetry = (() => {
  const clone = value => JSON.parse(JSON.stringify(value));
  const round = (value, digits = 1) => Number(value.toFixed(digits));
  const clamp = (value, max) => Math.max(0, Math.min(max, value));
  function sample(baseline, tick) {
    const next = clone(baseline);
    const wave = offset => Math.sin(tick / 4 + offset) * .065 + Math.sin(tick / 1.7 + offset) * .025;
    const load = 1 + wave(0), wait = 1 + wave(2) * 2;
    const m = next.summary.metrics;
    // Only gauges vary. Completed phase timings, counters and configuration remain stable.
    Object.entries(m).forEach(([key, value], index) => {
      if (/(_total|_peak_bytes|checkpoint_|_load_time_seconds|evaluation_time|capacity_used)/.test(key)) return;
      const factor = /wait|queue|pending|latency|straggler|bubble|delay|first_token/.test(key) ? wait : load;
      let result = value * (/memory|clock|temperature|loss/.test(key) ? 1 + wave(index) * .2 : factor);
      if (key.endsWith('_ratio')) result = clamp(result, 1);
      if (key.endsWith('_percent')) result = clamp(result, 100);
      m[key] = round(result, /queued|pending|turns|queue_depth/.test(key) ? 0 : 3);
    });
    m.training_step_time_seconds_p95 = round(baseline.summary.metrics.training_step_time_seconds_p95 / load, 3);
    next.summary.diagnostic.collective_seconds = m.training_communication_seconds_p95;
    next.resources.nodes.forEach((node, i) => {
      const base = baseline.resources.nodes[i], factor = 1 + wave(i * .7);
      node.cpu_utilization_percent = round(clamp(base.cpu_utilization_percent * factor, 100));
      node.memory_used_gib = round(clamp(base.memory_used_gib * (1 + wave(i) * .15), node.memory_total_gib));
      node.nic_receive_gbps = round(base.nic_receive_gbps * load);
      node.nic_transmit_gbps = round(base.nic_transmit_gbps * load);
      node.gpus.forEach((gpu, j) => {
        const original = base.gpus[j], activity = 1 + wave(i + j * .2);
        gpu.utilization_percent = round(clamp(original.utilization_percent * activity, 100));
        gpu.memory_used_gib = round(original.memory_used_gib * (1 + wave(i + j) * .1));
        gpu.power_watts = Math.round(original.power_watts * activity);
        gpu.temperature_celsius = round(original.temperature_celsius + wave(i + j) * 20);
        gpu.clock_mhz = Math.round(original.clock_mhz * (1 + wave(i + j) * .1));
      });
    });
    const nodes = next.resources.nodes, gpus = nodes.flatMap(node => node.gpus);
    const sum = (items, key) => items.reduce((total, item) => total + item[key], 0);
    m.host_cpu_utilization_percent = round(sum(nodes, 'cpu_utilization_percent') / nodes.length);
    m.host_memory_used_gib = round(sum(nodes, 'memory_used_gib'));
    m.network_transmit_gbps = round(sum(nodes, 'nic_transmit_gbps'));
    m.network_receive_gbps = round(sum(nodes, 'nic_receive_gbps'));
    for (const [metric, field] of [['gpu_utilization_percent', 'utilization_percent'], ['gpu_memory_used_gib', 'memory_used_gib'], ['gpu_power_watts', 'power_watts'], ['gpu_temperature_celsius', 'temperature_celsius'], ['gpu_clock_mhz', 'clock_mhz']]) m[metric] = round(sum(gpus, field) / gpus.length);
    next.storage.devices.forEach((device, i) => {
      const base = baseline.storage.devices[i], factor = 1 + wave(i) * 1.5;
      for (const key of ['read_gbps', 'write_gbps', 'iops', 'latency_ms_p95', 'queue_depth', 'utilization_percent']) {
        device[key] = round(base[key] * factor, key === 'queue_depth' ? 0 : 2);
      }
      device.utilization_percent = clamp(device.utilization_percent, 100);
    });
    m.storage_latency_ms_p95 = Math.max(...next.storage.devices.map(device => device.latency_ms_p95));
    for (const block of Object.values(next.interconnect)) {
      if (!block || !block.labels) continue;
      for (const key of ['throughput_gbps', 'latency_us', 'utilization_percent']) {
        block[key] = block[key].map(row => row.map(value => value === null ? null : round(clamp(value * (key === 'latency_us' ? wait : load), key === 'utilization_percent' ? 100 : Infinity))));
      }
    }
    next.dataMovement.paths.forEach(path => {
      path.data_movement_duration_seconds = round(path.data_movement_duration_seconds / load, 3);
      path.data_movement_effective_bandwidth_bytes_per_second = Math.round(path.data_movement_bytes_total / path.data_movement_duration_seconds);
      path.baseline_utilization_ratio = path.data_movement_effective_bandwidth_bytes_per_second / path.baseline_bandwidth_bytes_per_second;
    });
    return next;
  }
  return { sample };
})();
if (typeof module !== 'undefined') module.exports = DemoTelemetry;
