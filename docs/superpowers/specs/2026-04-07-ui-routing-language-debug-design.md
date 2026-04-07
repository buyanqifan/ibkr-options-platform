## Summary

Fix three UX/debugging regressions in the Dash shell:

1. language selection must persist across page navigation and browser refresh
2. the mobile navbar toggler must actually open and close the collapsed menu
3. page render failures must show full traceback details in-page for personal debugging

The implementation stays focused on the app shell and routing layer. No strategy, backtest, or page business logic changes are included.

## Goals

- Persist the user's chosen language across route changes and refreshes.
- Keep navbar language state and visible labels synchronized.
- Make the mobile navbar toggler functional.
- Preserve the current manual routing approach.
- Show complete exception information in the page body when a route fails to render.
- Keep the existing non-blocking loading behavior intact.

## Non-Goals

- No server-side language persistence via cookies, database, or session storage.
- No i18n redesign for all page content.
- No navigation auto-close when clicking a menu item.
- No changes to trading, backtesting, or `binbin_god` runtime behavior.

## Current Problems

### Language state resets

The app shell defines `language-store`, but the current navbar does not read from it. The dropdown is hard-coded to `en`, and rebuilding the navbar content recreates the selector with the default language. Because navigation now uses full-page links, users lose their language choice on route changes and refresh.

### Mobile navbar toggler is incomplete

The layout includes `NavbarToggler` and `Collapse`, but there is no callback updating `navbar-collapse.is_open`. On smaller screens, the hamburger control does not actually drive the menu.

### Render failures are under-exposed

The current manual router catches exceptions and renders a friendly alert with `str(exc)`, but it hides the traceback. For a personal debugging tool, that makes diagnosis slower than necessary.

## Recommended Approach

Use browser-local state for language persistence, wire navbar rendering from that state, add a small toggler callback, and upgrade route error rendering to include full traceback details.

This approach is recommended because it directly fixes the observed regressions with the least architectural churn. It avoids cookie/session complexity while still satisfying cross-page and refresh persistence.

## Design

### 1. Persist language in browser local storage

Update the shell store in `app/layout.py`:

- change `dcc.Store(id="language-store", data="en")` to use `storage_type="local"`

Effect:

- language survives page refresh
- language survives route changes
- language usually survives browser restart because it is stored in browser local storage

This does not make the server aware of the preferred language during first render. That tradeoff is acceptable for this debugging-oriented app shell.

### 2. Make navbar state derive from the language store

Update `app/components/navbar.py` so the language selector is not hard-coded to English.

Changes:

- `create_navbar_items()` remains the source for translated nav labels
- the selector's value is driven by `language-store`
- navbar content rebuilds from the persisted language value rather than from a fixed default

Implementation shape:

- keep a callback that renders the navbar children from the selected language
- add a callback/output path so the selector value also reflects the persisted store value on load

Result:

- users can choose Chinese once and keep it across refreshes and route changes
- the selector and menu labels stay in sync

### 3. Add a real mobile toggler callback

Add a callback in `app/components/navbar.py`:

- input: `navbar-toggler.n_clicks`
- state: `navbar-collapse.is_open`
- output: `navbar-collapse.is_open`

Behavior:

- each click toggles open/closed state
- no automatic collapse on navigation

This is intentionally minimal and matches the requested behavior.

### 4. Show full traceback in route render failures

Update `display_page()` in `app/layout.py` to render full exception details.

Changes:

- catch exceptions around `_render_page(pathname)` as today
- include:
  - exception type
  - string message
  - full traceback text
- render traceback in a readable `html.Pre` or equivalent scrollable block

Result:

- the page stays visible instead of hard-failing
- debugging information is complete enough for personal use

### 5. Keep existing routing and loading behavior

Do not change:

- full-page navigation through app routes
- manual `_ROUTE_KEYS` resolution
- callable-layout support
- the current CSS that disables pointer-event blocking from Dash loading overlays

This keeps the recent route-interaction fixes intact while addressing the remaining gaps.

## Files To Change

- `app/layout.py`
- `app/components/navbar.py`
- `tests/unit/test_layout_routing.py`
- `tests/unit/test_loading_interaction.py`
- optional new test file if navbar interaction assertions are cleaner there

## Testing Strategy

### Routing tests

Add or update tests to verify:

- unknown routes still return a 404 view
- callable page layouts still render correctly
- page render failures include full traceback text in the returned component tree

### Language persistence tests

Add or update tests to verify:

- `language-store` is configured with `storage_type="local"`
- navbar language callbacks respect persisted language values
- selector value is not forced back to `en` after navbar regeneration

### Navbar interaction tests

Add tests to verify:

- `navbar-toggler` toggles `navbar-collapse.is_open`
- no regressions to the non-blocking loading wrapper behavior

## Risks And Mitigations

### Risk: duplicated callback ownership around navbar children/value

Mitigation:

- keep callback responsibilities narrow
- use one callback for toggle state and one for language-driven navbar rendering/value sync
- avoid overlapping outputs unless required and explicitly supported

### Risk: traceback rendering becomes unreadable

Mitigation:

- render traceback in a monospace block with scrolling
- keep the summary line above it short and clear

### Risk: local storage introduces a brief first-render mismatch

Mitigation:

- accept this tradeoff explicitly
- prioritize correctness after hydration over adding cookie/session complexity

## Success Criteria

- Changing language to Chinese persists after navigation and refresh.
- Mobile navbar toggler opens and closes the collapsed nav menu.
- A route render exception shows full traceback information in the page body.
- Existing routing and loading interaction tests remain green.
