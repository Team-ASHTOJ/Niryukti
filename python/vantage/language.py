"""Offline controlled-English modelling language.

Translates planner-style statements into a NIRYUKTI JSON model with a deterministic grammar
(no language model, no network, no third-party packages). Every accepted line is echoed back
in canonical algebra so the user can confirm the meaning; any line that does not match the
grammar is reported with its line number instead of being guessed.

Example
-------
    variables crude_a, crude_b between 0 and 300
    binary open_unit
    property sulfur: crude_a 3, crude_b 1
    maximize 9 crude_a + 15 crude_b - 50 open_unit
    capacity: crude_a + crude_b is at most 300
    spec: average sulfur of crude_a, crude_b at most 2
    link: crude_a + crude_b <= 300 open_unit
"""
import json
import re

NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?'
NAME = r'[A-Za-z_][A-Za-z0-9_]*'
_COMPARATORS = [
    (r'(?:is\s+)?(?:at\s+most|no\s+more\s+than|not\s+more\s+than|less\s+than\s+or\s+equal\s+to|'
     r'must\s+not\s+exceed|does\s+not\s+exceed|cannot\s+exceed|up\s+to|<=|=<|≤)', '<='),
    (r'(?:is\s+)?(?:at\s+least|no\s+less\s+than|not\s+less\s+than|greater\s+than\s+or\s+equal\s+to|'
     r'must\s+be\s+at\s+least|>=|=>|≥)', '>='),
    (r'(?:is\s+)?(?:equal\s+to|equals|exactly|must\s+equal|==|=)', '='),
]
_KEYWORDS = {'and', 'or', 'of', 'the', 'is', 'be', 'between', 'from', 'to', 'at', 'most',
             'least', 'sum', 'total', 'average'}


class TranslationError(ValueError):
    def __init__(self, errors):
        super().__init__('\n'.join(f'line {n}: {message}' for n, message in errors))
        self.errors = errors


def _clean(text):
    text = text.replace('$', '').replace('₹', '').replace('%', ' percent')
    text = re.sub(r'(?<=\d),(?=\d{3}\b)', '', text)  # 1,000 -> 1000
    return re.sub(r'\s+', ' ', text).strip()


def _number(text):
    text = text.strip()
    if not re.fullmatch(NUMBER, text, re.I):
        raise ValueError(f'expected a number, found "{text}"')
    return float(text)


def _names(text):
    items = [t for t in re.split(r'\s*,\s*|\s+and\s+', text.strip()) if t]
    for item in items:
        if not re.fullmatch(NAME, item):
            raise ValueError(f'"{item}" is not a valid variable name')
    return items


class Translator:
    def __init__(self):
        self.variables = {}      # name -> dict(lb, ub, type)
        self.order = []
        self.properties = {}     # property -> {variable: value}
        self.objective = None    # (sense, {name: coef}, offset)
        self.constraints = []    # dict(name, coefficients, lb, ub)
        self.echo = []
        self.warnings = []

    # -- declarations -------------------------------------------------------------------
    def _declare(self, name, lb=0.0, ub=None, kind='continuous', implicit=False):
        if name in _KEYWORDS:
            raise ValueError(f'"{name}" is a reserved word')
        if name not in self.variables:
            self.variables[name] = dict(lb=lb, ub=ub, type=kind)
            self.order.append(name)
            if implicit:
                self.warnings.append(f'"{name}" was not declared; assumed continuous and >= 0')
        elif not implicit:
            self.variables[name].update(lb=lb, ub=ub, type=kind)

    # -- expressions --------------------------------------------------------------------
    def expression(self, text):
        """Linear expression -> ({name: coefficient}, constant)."""
        text = text.strip()
        m = re.fullmatch(r'(?:the\s+)?(?:sum|total)\s+of\s+(.+)', text, re.I)
        if m:
            return {name: 1.0 for name in _names(m.group(1))}, 0.0
        m = re.fullmatch(r'average\s+(' + NAME + r')\s+of\s+(.+)', text, re.I)
        if m:
            raise ValueError('averages are only valid in a comparison ("average P of a, b at most 2")')
        terms, constant = {}, 0.0
        text = re.sub(r'\s*([+-])\s*', r' \1 ', ' ' + text).strip()
        if not text.startswith(('+', '-')):
            text = '+ ' + text
        for sign, body in re.findall(r'([+-])\s+((?:(?![+-]\s).)+)', text + ' '):
            body = body.strip().replace('*', ' ')
            factor = -1.0 if sign == '-' else 1.0
            m = re.fullmatch(r'(' + NUMBER + r')?\s*(' + NAME + r')?', body, re.I)
            if not m or (m.group(1) is None and m.group(2) is None):
                raise ValueError(f'cannot read the term "{body}"')
            coefficient = float(m.group(1)) if m.group(1) else 1.0
            if m.group(2):
                name = m.group(2)
                self._declare(name, implicit=True)
                terms[name] = terms.get(name, 0.0) + factor * coefficient
            else:
                constant += factor * coefficient
        return terms, constant

    # -- statements ---------------------------------------------------------------------
    def statement(self, line):
        lowered = line.lower()
        # Variable declarations.
        m = re.fullmatch(r'(binary|integer|continuous)?\s*(?:variables?|decisions?)?\s*(.+?)'
                         r'\s+(?:between|from)\s+(' + NUMBER + r')\s+(?:and|to)\s+(' + NUMBER + r')',
                         line, re.I)
        if m and ':' not in line and not re.search(r'<=|>=|=|at most|at least|\bis\b', lowered):
            kind = (m.group(1) or 'continuous').lower()
            lb, ub = float(m.group(3)), float(m.group(4))
            if lb > ub:
                raise ValueError('lower bound exceeds upper bound')
            names = _names(m.group(2))
            for name in names:
                self._declare(name, lb, ub, kind)
            self.echo.append(f'{kind} {", ".join(names)} in [{lb:g}, {ub:g}]')
            return
        m = re.fullmatch(r'(binary|integer|continuous|free)\s+(?:variables?\s+)?(.+)', line, re.I)
        if m:
            kind = m.group(1).lower()
            names = _names(m.group(2))
            for name in names:
                if kind == 'binary':
                    self._declare(name, 0, 1, 'binary')
                elif kind == 'free':
                    self._declare(name, None, None, 'continuous')
                else:
                    self._declare(name, 0, None, kind)
            self.echo.append(f'{kind} {", ".join(names)}')
            return
        m = re.fullmatch(r'(?:variables?|decisions?)\s+(.+)', line, re.I)
        if m:
            names = _names(m.group(1))
            for name in names:
                self._declare(name)
            self.echo.append(f'continuous {", ".join(names)} >= 0')
            return
        # Properties (qualities) for weighted-average specifications.
        m = re.fullmatch(r'property\s+(' + NAME + r')\s*:\s*(.+)', line, re.I)
        if m:
            values = {}
            for item in re.split(r'\s*,\s*', m.group(2)):
                pair = re.fullmatch(r'(' + NAME + r')\s+(?:is\s+|=\s*|has\s+)?(' + NUMBER + r')', item.strip(), re.I)
                if not pair:
                    raise ValueError(f'cannot read property entry "{item}"; use "name value"')
                values[pair.group(1)] = float(pair.group(2))
                self._declare(pair.group(1), implicit=True)
            self.properties.setdefault(m.group(1).lower(), {}).update(values)
            self.echo.append(f'property {m.group(1).lower()}: ' +
                             ', '.join(f'{k}={v:g}' for k, v in values.items()))
            return
        # Objective.
        m = re.fullmatch(r'(minimi[sz]e|maximi[sz]e|min|max)\s+(?:' + NAME + r'\s*:\s*)?(.+)', line, re.I)
        if m:
            if self.objective is not None:
                raise ValueError('a second objective was given')
            sense = 'max' if m.group(1).lower().startswith('max') else 'min'
            terms, constant = self.expression(m.group(2))
            self.objective = (sense, terms, constant)
            self.echo.append(f'{sense}imize {self._format(terms, constant)}')
            return
        # Constraints: optional "name:" prefix.
        name = None
        m = re.fullmatch(r'(' + NAME + r')\s*:\s*(.+)', line)
        if m:
            name, line = m.group(1), m.group(2)
        m = re.fullmatch(r'(.+?)\s+(?:is\s+)?between\s+(' + NUMBER + r')\s+and\s+(' + NUMBER + r')', line, re.I)
        if m:
            terms, constant = self.expression(m.group(1))
            lb, ub = float(m.group(2)) - constant, float(m.group(3)) - constant
            self._add(name, terms, lb, ub)
            return
        for pattern, op in _COMPARATORS:
            m = re.fullmatch(r'(.+?)\s*' + pattern + r'\s*(.+)', line, re.I)
            if not m:
                continue
            left, right = m.group(1), m.group(2)
            avg = re.fullmatch(r'(?:the\s+)?average\s+(' + NAME + r')\s+of\s+(.+)', left.strip(), re.I)
            if avg:
                prop = avg.group(1).lower()
                if prop not in self.properties:
                    raise ValueError(f'property "{prop}" has not been defined')
                limit = _number(re.sub(r'\s*percent$', '', right.strip(), flags=re.I))
                terms = {}
                for var in _names(avg.group(2)):
                    if var not in self.properties[prop]:
                        raise ValueError(f'"{var}" has no {prop} value')
                    self._declare(var, implicit=True)
                    terms[var] = self.properties[prop][var] - limit
                # sum q_i x_i (op) limit * sum x_i  <=>  sum (q_i - limit) x_i (op) 0
                self._add(name, terms, *self._bounds(op, 0.0))
                return
            lt, lc = self.expression(left)
            rt, rc = self.expression(re.sub(r'\s*percent$', '', right.strip(), flags=re.I))
            terms = dict(lt)
            for var, coef in rt.items():
                terms[var] = terms.get(var, 0.0) - coef
            self._add(name, terms, *self._bounds(op, rc - lc))
            return
        raise ValueError('not recognised; see the grammar in the module documentation')

    @staticmethod
    def _bounds(op, rhs):
        return {'<=': (None, rhs), '>=': (rhs, None), '=': (rhs, rhs)}[op]

    def _add(self, name, terms, lb, ub):
        terms = {k: v for k, v in terms.items() if v != 0}
        if not terms:
            raise ValueError('constraint has no variables')
        name = name or f'r{len(self.constraints)}'
        if any(c['name'] == name for c in self.constraints):
            raise ValueError(f'duplicate constraint name "{name}"')
        self.constraints.append(dict(name=name, coefficients=terms, lb=lb, ub=ub))
        if lb is not None and ub is not None and lb == ub:
            text = f'{self._format(terms)} = {lb:g}'
        elif lb is not None and ub is not None:
            text = f'{lb:g} <= {self._format(terms)} <= {ub:g}'
        elif lb is not None:
            text = f'{self._format(terms)} >= {lb:g}'
        else:
            text = f'{self._format(terms)} <= {ub:g}'
        self.echo.append(f'{name}: {text}')

    @staticmethod
    def _format(terms, constant=0.0):
        parts = []
        for var, coef in terms.items():
            sign = '-' if coef < 0 else '+'
            magnitude = abs(coef)
            body = var if magnitude == 1 else f'{magnitude:g} {var}'
            parts.append(f'{sign} {body}')
        if constant:
            parts.append(f'{"-" if constant < 0 else "+"} {abs(constant):g}')
        text = ' '.join(parts) or '0'
        return text[2:] if text.startswith('+ ') else '-' + text[2:]

    def model(self, name):
        if self.objective is None:
            raise TranslationError([(0, 'no objective ("minimize ..." or "maximize ...") was given')])
        sense, terms, offset = self.objective
        variables = []
        for var in self.order:
            spec = self.variables[var]
            entry = dict(name=var, lb=spec['lb'] if spec['lb'] is not None else '-inf', ub=spec['ub'])
            if spec['type'] != 'continuous':
                entry['type'] = spec['type']
            variables.append(entry)
        return dict(name=name, sense=sense, variables=variables,
                    objective=dict(linear=[terms.get(v, 0.0) for v in self.order], offset=offset),
                    constraints=self.constraints)


def translate(text, *, name='translated_model'):
    """Return (model_dict, report). Raises TranslationError listing every unparsed line."""
    translator, errors = Translator(), []
    for number, raw in enumerate(text.splitlines(), 1):
        line = _clean(raw.split('#', 1)[0])
        if not line:
            continue
        try:
            translator.statement(line.rstrip('.'))
        except ValueError as error:
            errors.append((number, f'{error} — "{raw.strip()}"'))
    if errors:
        raise TranslationError(errors)
    model = translator.model(name)
    report = dict(understood=translator.echo, warnings=translator.warnings,
                  variables=len(model['variables']), constraints=len(model['constraints']),
                  scope='Deterministic controlled-English grammar; not a language model. '
                        'Review the echoed algebra before solving.')
    return model, report


def translate_file(path, output=None, *, name=None):
    with open(path) as handle:
        model, report = translate(handle.read(), name=name or re.sub(r'\W+', '_', str(path).rsplit('/', 1)[-1].rsplit('.', 1)[0]))
    if output:
        with open(output, 'w') as handle:
            json.dump(model, handle, indent=2)
            handle.write('\n')
    return model, report
