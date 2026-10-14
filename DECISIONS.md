# DECISIONS.md

1. **Indexes.** Unique `emp_code` on employees and unique `(emp_code, date)` on attendance: they enforce "one per key" and make creates and punch-ins race-safe. `(date desc, emp_code)` and `(status, date desc, emp_code)` serve the attendance list and the month scans; `(emp_code, punch_in desc)` serves punch-out. Three employee indexes on `department` and `joined_on` serve listing, headcount and trend. I rejected copying `department` onto attendance records to avoid a join: it would go stale.

2. **Punch-in race.** Both requests try to insert the same `(emp_code, date)`. The unique index lets MongoDB accept exactly one; the other gets a duplicate-key error, which I turn into a 409. There is no check-then-insert.

3. **Ties.** Tied employees share a rank (1, 2, 2, 4) and I return every row whose rank is within `limit`, so a tie at the cutoff returns more than `limit` rows. Cutting by position would drop one of them arbitrarily.

4. **Headcount.** It comes from its own query on employees (joined on or before month end), merged with the activity totals, so people with no logs still count.

5. **100x data.** Replace skip/limit paging with keyset paging, and keep precomputed daily totals so analytics stop scanning raw records.