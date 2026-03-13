-- ══════════════════════════════════════════
-- setup_db.sql  — Create the energy table
-- ══════════════════════════════════════════
CREATE TABLE IF NOT EXISTS energy (
    datetime     TEXT PRIMARY KEY,
    consumption  REAL,
    hour         INTEGER,
    day_of_week  INTEGER,
    month        INTEGER,
    year         INTEGER,
    is_weekend   INTEGER,
    quarter      INTEGER
);

-- ══════════════════════════════════════════
-- peak_demand.sql  — Find peak hours
-- ══════════════════════════════════════════

-- Top 10 highest consumption hours ever
SELECT datetime, consumption
FROM energy
ORDER BY consumption DESC
LIMIT 10;

-- Average consumption by hour of day
SELECT hour,
       ROUND(AVG(consumption), 2) AS avg_consumption,
       ROUND(MAX(consumption), 2) AS max_consumption
FROM energy
GROUP BY hour
ORDER BY hour;

-- Peak vs off-peak comparison
SELECT
    CASE WHEN hour BETWEEN 7 AND 22 THEN 'Peak (7am-10pm)'
         ELSE 'Off-Peak'
    END AS period,
    ROUND(AVG(consumption), 2) AS avg_consumption
FROM energy
GROUP BY period;

-- ══════════════════════════════════════════
-- seasonal_analysis.sql — Seasonality
-- ══════════════════════════════════════════

-- Average consumption by month (seasonality)
SELECT month,
       ROUND(AVG(consumption), 2) AS avg_consumption
FROM energy
GROUP BY month
ORDER BY month;

-- Weekday vs weekend
SELECT
    CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    ROUND(AVG(consumption), 2) AS avg_consumption
FROM energy
GROUP BY is_weekend;

-- Year-over-year growth
SELECT year,
       ROUND(AVG(consumption), 2) AS avg_consumption,
       ROUND(SUM(consumption), 0) AS total_consumption
FROM energy
GROUP BY year
ORDER BY year;

-- Quarter breakdown
SELECT year, quarter,
       ROUND(AVG(consumption), 2) AS avg_consumption
FROM energy
GROUP BY year, quarter
ORDER BY year, quarter;
