/* global $, d3 */

(function () {
    "use strict";

    var POLL_MS = 3000;
    var COLORS = [
        "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6",
        "#1ABC9C", "#E67E22", "#3498DB", "#E84393",
        "#00CEC9", "#FD79A8"
    ];

    // Graph layout constants
    var TRUNK_X = 60;
    var TRUNK_Y = 40;
    var LANE_H = 70;
    var COMMIT_DX = 160;
    var NODE_R = 9;
    var TRUNK_R = 14;

    var prevData = null;
    var tooltip = null;

    // ---- Init ----

    $(function () {
        tooltip = d3.select("body").append("div").attr("class", "commit-tooltip");
        fetchStatus();
        pollBranches();
        setInterval(pollBranches, POLL_MS);

        $("#preset-select").on("change", function () {
            var v = $(this).val();
            if (v) $("#question-input").val(v);
        });

        $("#ask-btn").on("click", submitQuery);
        $("#question-input").on("keydown", function (e) {
            if (e.key === "Enter") submitQuery();
        });
        $("#cache-toggle").on("change", function () {
            $(this).next(".toggle-text").text(this.checked ? "On" : "Off");
        });
    });

    // ---- Status ----

    function fetchStatus() {
        $.getJSON("/api/status", function (data) {
            $("#status-line").text(
                "Trunk: " + data.demo_main +
                " \u00b7 HEAD: " + data.head.short_hash +
                " \u00b7 Last commit: " + data.head.message
            );
        });
    }

    // ---- Branch graph ----

    function pollBranches() {
        $.getJSON("/api/branches", function (data) {
            renderGraph(data);
        });
    }

    function computeGraphSize(branches, containerW) {
        var maxCommits = 0;
        branches.forEach(function (b) {
            if (b.commits.length > maxCommits) maxCommits = b.commits.length;
        });
        var neededW = TRUNK_X + (maxCommits + 2) * COMMIT_DX + 200;
        return {
            w: Math.max(containerW, neededW),
            h: Math.max(180, TRUNK_Y + (branches.length + 1) * LANE_H + 30)
        };
    }

    function drawTrunk(svg, trunkHash) {
        var trunkG = svg.selectAll(".trunk-g").data([trunkHash]);
        var trunkEnter = trunkG.enter().append("g").attr("class", "trunk-g");
        trunkEnter.append("circle")
            .attr("class", "trunk-node")
            .attr("r", 0)
            .attr("fill", "#4A90D9")
            .attr("stroke", "#fff")
            .attr("stroke-width", 2.5);
        trunkEnter.append("text")
            .attr("text-anchor", "middle")
            .attr("dy", -22)
            .attr("fill", "#fff")
            .attr("font-family", "SF Mono, Fira Code, Consolas, monospace")
            .attr("font-size", "13px")
            .attr("font-weight", "600")
            .text("main");

        var trunkMerge = trunkG.merge(trunkEnter);
        trunkMerge.attr("transform", "translate(" + TRUNK_X + "," + TRUNK_Y + ")");
        trunkMerge.select("circle").transition().duration(500).attr("r", TRUNK_R);
    }

    function drawBranchConnector(g, laneY, color, idx) {
        var firstCX = TRUNK_X + COMMIT_DX;
        var connectorData = [
            { x: TRUNK_X, y: TRUNK_Y },
            { x: TRUNK_X + COMMIT_DX * 0.4, y: TRUNK_Y + (laneY - TRUNK_Y) * 0.5 },
            { x: firstCX, y: laneY }
        ];

        var line = d3.line()
            .x(function (d) { return d.x; })
            .y(function (d) { return d.y; })
            .curve(d3.curveBasis);

        var connector = g.selectAll(".connector-line").data([1]);
        var connectorEnter = connector.enter().append("path")
            .attr("class", "connector-line")
            .attr("d", line(connectorData))
            .attr("stroke-dasharray", function () {
                return this.getTotalLength();
            })
            .attr("stroke-dashoffset", function () {
                return this.getTotalLength();
            });
        connectorEnter.transition().duration(600).delay(idx * 80)
            .attr("stroke-dashoffset", 0);
        connector.merge(connectorEnter)
            .attr("stroke", color)
            .attr("d", line(connectorData));
    }

    function drawBranchLine(g, commits, laneY, color, idx) {
        if (commits.length <= 1) return;
        var firstCX = TRUNK_X + COMMIT_DX;
        var lastCX = TRUNK_X + COMMIT_DX * commits.length;

        var branchLine = g.selectAll(".branch-line").data([1]);
        var branchLineEnter = branchLine.enter().append("line")
            .attr("class", "branch-line")
            .attr("x1", firstCX).attr("y1", laneY)
            .attr("x2", firstCX).attr("y2", laneY);
        branchLineEnter.transition().duration(500).delay(idx * 80 + 400)
            .attr("x2", lastCX);
        branchLine.merge(branchLineEnter)
            .attr("stroke", color)
            .transition().duration(400)
            .attr("x1", firstCX).attr("y1", laneY)
            .attr("x2", lastCX).attr("y2", laneY);
    }

    function drawCommitNodes(g, commits, laneY, color, idx) {
        var commitG = g.selectAll(".commit-g")
            .data(commits, function (d) { return d.hash; });

        commitG.exit()
            .transition().duration(200).style("opacity", 0).remove();

        var commitEnter = commitG.enter().append("g")
            .attr("class", "commit-g")
            .style("opacity", 0);

        commitEnter.append("circle")
            .attr("class", "commit-node")
            .attr("r", 0)
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5);

        commitEnter.append("text")
            .attr("class", "commit-label")
            .attr("text-anchor", "middle")
            .attr("dy", -16);

        var commitMerge = commitEnter.merge(commitG);
        commitMerge.select("circle").attr("fill", color);
        commitMerge.select("text.commit-label").attr("fill", color);

        commitMerge.each(function (cd, ci) {
            var cg = d3.select(this);
            var cx = TRUNK_X + COMMIT_DX * (ci + 1);
            var delay = idx * 80 + ci * 150 + 300;

            cg.transition().duration(400).delay(delay)
                .attr("transform", "translate(" + cx + "," + laneY + ")")
                .style("opacity", 1);

            cg.select("circle")
                .transition().duration(300).delay(delay)
                .attr("r", NODE_R);

            cg.select("text").text(cd.label);

            cg.on("mouseenter", function (event) {
                tooltip
                    .style("opacity", 1)
                    .html(
                        "<strong>" + cd.label + "</strong><br>" +
                        cd.hash + "<br>" +
                        cd.message + "<br>" +
                        cd.date
                    )
                    .style("left", (event.pageX + 12) + "px")
                    .style("top", (event.pageY - 10) + "px");
            }).on("mouseleave", function () {
                tooltip.style("opacity", 0);
            });
        });
    }

    function drawBranchLabel(g, branchData, laneY, color, idx) {
        var labelX = TRUNK_X + COMMIT_DX * (Math.max(branchData.commits.length, 1) + 0.5);
        var labelEl = g.selectAll(".branch-label").data([branchData.label]);
        var labelEnter = labelEl.enter().append("text")
            .attr("class", "branch-label")
            .attr("x", labelX)
            .attr("y", laneY + 5)
            .style("opacity", 0)
            .text(branchData.label);
        labelEnter.transition().duration(400).delay(idx * 80 + 600)
            .style("opacity", 1);
        labelEl.merge(labelEnter)
            .attr("fill", color)
            .transition().duration(300)
            .attr("x", labelX)
            .attr("y", laneY + 5)
            .text(branchData.label);
    }

    function renderGraph(data) {
        var branches = data.branches || [];
        var svgEl = document.getElementById("branch-graph");
        var containerW = svgEl.parentElement.clientWidth - 48;
        var size = computeGraphSize(branches, containerW);

        var svg = d3.select("#branch-graph")
            .attr("width", size.w)
            .attr("height", size.h);

        drawTrunk(svg, data.trunk);

        var colorMap = {};
        branches.forEach(function (b, i) {
            colorMap[b.name] = COLORS[i % COLORS.length];
        });

        var branchG = svg.selectAll(".branch-g")
            .data(branches, function (d) { return d.name; });

        branchG.exit()
            .transition().duration(300).style("opacity", 0).remove();

        var branchEnter = branchG.enter().append("g")
            .attr("class", "branch-g")
            .style("opacity", 0);

        var branchMerge = branchEnter.merge(branchG);

        branchMerge.each(function (branchData, idx) {
            var g = d3.select(this);
            var color = colorMap[branchData.name];
            var laneY = TRUNK_Y + (idx + 1) * LANE_H;

            drawBranchConnector(g, laneY, color, idx);
            drawBranchLine(g, branchData.commits, laneY, color, idx);
            drawCommitNodes(g, branchData.commits, laneY, color, idx);
            drawBranchLabel(g, branchData, laneY, color, idx);
        });

        branchMerge
            .transition().duration(400)
            .style("opacity", 1);

        svg.select(".trunk-g").raise();

        prevData = data;
    }

    // ---- Query ----

    function submitQuery() {
        var question = $.trim($("#question-input").val());
        if (!question) return;

        var engine = $("#engine-select").val();
        var $btn = $("#ask-btn");
        var $result = $("#query-result");

        $btn.prop("disabled", true).text("Thinking...");
        $result.removeClass("hidden").html(
            '<div><span class="spinner"></span> Translating to SQL and querying all branches...</div>'
        );

        $.ajax({
            url: "/api/query",
            method: "POST",
            contentType: "application/json",
            data: JSON.stringify({
                question: question,
                engine: engine,
                use_cache: $("#cache-toggle").is(":checked")
            }),
            success: function (data) {
                $btn.prop("disabled", false).text("Ask");
                renderResult(data);
            },
            error: function () {
                $btn.prop("disabled", false).text("Ask");
                $result.html('<div class="error-box">Request failed. Check the server logs.</div>');
            }
        });
    }

    function renderResult(data) {
        var $r = $("#query-result").removeClass("hidden").empty();

        if (data.error) {
            $r.append('<div class="error-box">' + escHtml(data.error) + '</div>');
            if (data.hint) {
                $r.append('<div class="hint-box">Try instead: ' + escHtml(data.hint) + '</div>');
            }
            if (data.sql) {
                $r.append(buildSqlDetails(data.sql));
            }
            return;
        }

        $r.append(buildSqlDetails(data.sql));
        var cacheTag = data.cache_hit
            ? ' \u00b7 <span class="cache-hit">CACHE HIT</span>'
            : ' \u00b7 <span class="cache-miss">LLM call</span>';
        var statsTag = '';
        if (data.stats) {
            statsTag = ' \u00b7 <span class="stats-tag">' +
                formatBytes(data.stats.result_bytes) +
                ' across ' + data.stats.branches_queried + ' branches</span>';
        }
        $r.append(
            '<div class="result-meta">Engine: <strong>' + escHtml(data.engine) +
            '</strong> \u00b7 Result type: <strong>' + escHtml(data.result_type) +
            '</strong>' + cacheTag + statsTag + '</div>'
        );

        if (!data.meta) {
            $r.append('<div class="error-box">No meta-answer available.</div>');
            return;
        }

        var meta = data.meta;

        if (meta.kind === "boolean") {
            renderBoolean($r, meta);
        } else if (meta.kind === "number") {
            renderNumber($r, meta);
        } else if (meta.kind === "set") {
            renderSet($r, meta);
        }
    }

    // ---- Boolean rendering: big box + pie chart if mixed ----

    function renderBoolean($r, meta) {
        var verdict = meta.details.verdict;
        var trueCount = meta.details.true_count || 0;
        var falseCount = meta.details.false_count || 0;

        if (verdict === "definitely_true") {
            $r.append(
                '<div class="answer-box answer-true">' +
                '<div class="answer-value">YES</div>' +
                '<div class="answer-label">All ' + (trueCount + falseCount) + ' branches agree</div></div>'
            );
        } else if (verdict === "definitely_false") {
            $r.append(
                '<div class="answer-box answer-false">' +
                '<div class="answer-value">NO</div>' +
                '<div class="answer-label">All ' + (trueCount + falseCount) + ' branches agree</div></div>'
            );
        } else {
            $r.append(
                '<div class="answer-box answer-mixed">' +
                '<div class="answer-value">UNCLEAR</div>' +
                '<div class="answer-label">Branches disagree (' + trueCount + ' true, ' + falseCount + ' false)</div></div>'
            );
            renderBooleanPie($r, meta);
        }

        renderBranchTable($r, meta);
    }

    function renderBooleanPie($r, meta) {
        var branches = Object.keys(meta.per_branch);
        var pieData = [
            { label: "True", value: meta.details.true_count, color: "#2ecc71" },
            { label: "False", value: meta.details.false_count, color: "#e74c3c" }
        ];

        var $container = $('<div class="pie-container"></div>');
        var svgId = "pie-" + Date.now();
        $container.append('<svg id="' + svgId + '" width="140" height="140"></svg>');

        var $legend = $('<div class="pie-legend"></div>');
        branches.forEach(function (b) {
            var val = meta.per_branch[b];
            var color = val ? "#2ecc71" : "#e74c3c";
            $legend.append(
                '<div class="pie-legend-item">' +
                '<span class="pie-legend-swatch" style="background:' + color + '"></span>' +
                escHtml(b) + ': ' + (val ? 'true' : 'false') +
                '</div>'
            );
        });
        $container.append($legend);
        $r.append($container);

        // Draw pie with D3
        var w = 140, h = 140, r = 60;
        var svg = d3.select("#" + svgId);
        var g = svg.append("g").attr("transform", "translate(" + w / 2 + "," + h / 2 + ")");
        var pie = d3.pie().value(function (d) { return d.value; }).sort(null);
        var arc = d3.arc().innerRadius(r * 0.5).outerRadius(r);
        g.selectAll("path").data(pie(pieData)).enter().append("path")
            .attr("d", arc)
            .attr("fill", function (d) { return d.data.color; })
            .attr("stroke", "#161b22")
            .attr("stroke-width", 2);
    }

    // ---- Number rendering: big box + bar chart if disagreement ----

    function renderNumber($r, meta) {
        var d = meta.details;

        if (d.agreement) {
            var fmtVal = typeof d.min === "number" ? formatNum(d.min) : d.min;
            $r.append(
                '<div class="answer-box answer-agree">' +
                '<div class="answer-value">' + escHtml(String(fmtVal)) + '</div>' +
                '<div class="answer-label">All branches agree</div></div>'
            );
        } else {
            $r.append(
                '<div class="answer-box answer-disagree">' +
                '<div class="answer-value">UNCLEAR</div>' +
                '<div class="answer-label">Range: ' + formatNum(d.min) + ' \u2013 ' + formatNum(d.max) + ' (mean ' + formatNum(d.mean) + ')</div></div>'
            );
            renderNumberChart($r, meta);
        }

        renderBranchTable($r, meta);
    }

    function formatNum(v) {
        if (typeof v !== "number") return String(v);
        if (Number.isInteger(v)) return v.toLocaleString();
        if (Math.abs(v) >= 1) return v.toFixed(2);
        return v.toFixed(4);
    }

    function renderNumberChart($r, meta) {
        var branches = Object.keys(meta.per_branch);
        var values = branches.map(function (b) { return { branch: b, value: meta.per_branch[b] }; });
        values.sort(function (a, b) { return a.value - b.value; });

        var chartW = Math.min(600, branches.length * 80 + 60);
        var chartH = 180;
        var margin = { top: 15, right: 20, bottom: 50, left: 60 };
        var innerW = chartW - margin.left - margin.right;
        var innerH = chartH - margin.top - margin.bottom;

        var $div = $('<div class="chart-container"></div>');
        var svgId = "num-chart-" + Date.now();
        $div.append('<svg id="' + svgId + '" width="' + chartW + '" height="' + chartH + '"></svg>');
        $r.append($div);

        var svg = d3.select("#" + svgId);
        var g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        var x = d3.scaleBand().domain(values.map(function (d) { return d.branch; }))
            .range([0, innerW]).padding(0.3);
        var y = d3.scaleLinear().domain([0, d3.max(values, function (d) { return d.value; }) * 1.1])
            .range([innerH, 0]);

        // Y axis
        g.append("g").attr("class", "axis")
            .call(d3.axisLeft(y).ticks(5).tickFormat(function (v) { return formatNum(v); }));

        // X axis with branch labels
        g.append("g").attr("class", "axis")
            .attr("transform", "translate(0," + innerH + ")")
            .call(d3.axisBottom(x))
            .selectAll("text")
            .attr("class", "branch-tick")
            .attr("transform", "rotate(-25)")
            .style("text-anchor", "end");

        // Bars
        g.selectAll(".bar").data(values).enter().append("rect")
            .attr("class", "bar")
            .attr("x", function (d) { return x(d.branch); })
            .attr("y", innerH)
            .attr("width", x.bandwidth())
            .attr("height", 0)
            .transition().duration(500)
            .attr("y", function (d) { return y(d.value); })
            .attr("height", function (d) { return innerH - y(d.value); });

        // Value labels on bars
        g.selectAll(".bar-label").data(values).enter().append("text")
            .attr("text-anchor", "middle")
            .attr("fill", "#ccc")
            .attr("font-size", "11px")
            .attr("font-family", "SF Mono, monospace")
            .attr("x", function (d) { return x(d.branch) + x.bandwidth() / 2; })
            .attr("y", function (d) { return y(d.value) - 4; })
            .text(function (d) { return formatNum(d.value); });
    }

    // ---- Set rendering: summary + git-diff-style branch comparison ----

    function renderSet($r, meta) {
        var d = meta.details;
        var intersection = d.intersection || [];
        var disagreement = d.disagreement || [];
        var total = (d.union || []).length;

        var nBranches = Object.keys(meta.per_branch).length;

        if (disagreement.length === 0) {
            $r.append(
                '<div class="answer-box answer-agree">' +
                '<div class="answer-value">' + intersection.length + ' items</div>' +
                '<div class="answer-label">All ' + nBranches + ' branches agree</div></div>'
            );
            renderItemList($r, intersection.slice(0, 10), "Consensus items");
        } else {
            $r.append(
                '<div class="answer-box answer-disagree">' +
                '<div class="answer-value">' + intersection.length + ' of ' + total + ' in consensus</div>' +
                '<div class="answer-label">Across all ' + nBranches + ' branches \u00b7 ' + disagreement.length + ' items where branches disagree</div></div>'
            );
            renderSetDiff($r, meta);
        }
    }

    function renderItemList($r, items, title) {
        var $section = $('<div class="diff-section"></div>');
        $section.append('<div class="diff-section-header diff-header-shared">' + escHtml(title) + '</div>');
        var html = items.map(function (it) {
            return '<span class="diff-item diff-item-shared">' + escHtml(String(it)) + '</span>';
        }).join("");
        $section.append('<div class="diff-items">' + (html || '<span class="diff-empty">None</span>') + '</div>');
        $r.append($section);
    }

    function renderSetDiff($r, meta) {
        var branches = Object.keys(meta.per_branch);
        if (branches.length < 2) return;

        var $controls = $('<div class="set-diff-controls"></div>');
        var $selA = $('<select id="diff-branch-a"></select>');
        var $selB = $('<select id="diff-branch-b"></select>');
        branches.forEach(function (b, i) {
            $selA.append('<option value="' + b + '"' + (i === 0 ? ' selected' : '') + '>' + escHtml(b) + '</option>');
            $selB.append('<option value="' + b + '"' + (i === 1 ? ' selected' : '') + '>' + escHtml(b) + '</option>');
        });
        $controls.append('<span>Compare:</span>').append($selA).append('<span>vs</span>').append($selB);
        $r.append($controls);

        var $diffView = $('<div id="diff-view"></div>');
        $r.append($diffView);

        function updateDiff() {
            var a = $selA.val();
            var b = $selB.val();
            var setA = new Set(meta.per_branch[a] || []);
            var setB = new Set(meta.per_branch[b] || []);

            var shared = [];
            var onlyA = [];
            var onlyB = [];

            setA.forEach(function (item) {
                if (setB.has(item)) shared.push(item);
                else onlyA.push(item);
            });
            setB.forEach(function (item) {
                if (!setA.has(item)) onlyB.push(item);
            });

            shared.sort();
            onlyA.sort();
            onlyB.sort();

            var pairTotal = shared.length + onlyA.length + onlyB.length;
            var pairPct = pairTotal > 0 ? Math.round(shared.length / pairTotal * 100) : 0;

            var $dv = $("#diff-view").empty();

            $dv.append(
                '<div class="result-meta" style="margin-bottom:0.8rem">' +
                'Pairwise: <strong>' + shared.length + ' of ' + pairTotal + '</strong> shared (' + pairPct + '% overlap)' +
                '</div>'
            );

            $dv.append(buildDiffSection("Shared (" + shared.length + ")", "shared", shared.slice(0, 10)));
            $dv.append(buildDiffSection("Only in " + a + " (" + onlyA.length + ")", "left", onlyA.slice(0, 10)));
            $dv.append(buildDiffSection("Only in " + b + " (" + onlyB.length + ")", "right", onlyB.slice(0, 10)));
        }

        $selA.on("change", updateDiff);
        $selB.on("change", updateDiff);
        updateDiff();
    }

    function buildDiffSection(title, kind, items) {
        var headerClass = kind === "shared" ? "diff-header-shared" :
                          kind === "left" ? "diff-header-left" : "diff-header-right";
        var itemClass = kind === "shared" ? "diff-item-shared" :
                        kind === "left" ? "diff-item-left" : "diff-item-right";

        var html = items.map(function (it) {
            return '<span class="diff-item ' + itemClass + '">' + escHtml(String(it)) + '</span>';
        }).join("");

        return '<div class="diff-section">' +
            '<div class="diff-section-header ' + headerClass + '">' + escHtml(title) + '</div>' +
            '<div class="diff-items">' + (html || '<span class="diff-empty">None</span>') + '</div></div>';
    }

    // ---- Shared: branch detail table ----

    function renderBranchTable($r, meta) {
        if (!meta.per_branch) return;
        var keys = Object.keys(meta.per_branch);
        if (!keys.length) return;

        var valueLabel = meta.kind === "boolean" ? "Result" :
                         meta.kind === "number" ? "Value" : "Count";

        var rows = keys.map(function (b) {
            var v = meta.per_branch[b];
            var display;
            if (Array.isArray(v)) display = v.length + " items";
            else if (typeof v === "boolean") display = v ? "true" : "false";
            else if (typeof v === "number") display = formatNum(v);
            else display = String(v);
            return '<tr><td>' + escHtml(b) + '</td><td>' + escHtml(display) + '</td></tr>';
        }).join("");

        $r.append(
            '<table class="branch-table"><thead><tr><th>Branch</th><th>' +
            valueLabel + '</th></tr></thead><tbody>' + rows + '</tbody></table>'
        );
    }

    function buildSqlDetails(sql) {
        return '<details class="sql-details">' +
            '<summary class="sql-summary">Generated SQL</summary>' +
            '<pre class="result-sql">' + escHtml(sql) + '</pre>' +
            '</details>';
    }

    function formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        var units = ['B', 'KB', 'MB', 'GB'];
        var i = Math.floor(Math.log(bytes) / Math.log(1024));
        i = Math.min(i, units.length - 1);
        var val = bytes / Math.pow(1024, i);
        return val.toFixed(i === 0 ? 0 : 1) + ' ' + units[i];
    }

    function escHtml(s) {
        var div = document.createElement("div");
        div.appendChild(document.createTextNode(s));
        return div.innerHTML;
    }

})();
