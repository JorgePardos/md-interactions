"""The interactive configuration builder."""

from __future__ import annotations

import pytest
import yaml

import MDAnalysis as mda

from md_interactions.config import Config, load_config
from md_interactions.exceptions import MDInteractionsError
from md_interactions.testing import make_toy_universe, write_toy_system
from md_interactions.wizard import AtomResolver, Prompt, run_wizard


class ScriptedPrompt(Prompt):
    """Prompt fed from a list of answers; records everything printed."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.output: list[str] = []
        super().__init__(self._next, self._record)

    def _next(self, question: str) -> str:
        if not self.answers:
            raise EOFError(f"no answers left at: {question!r}")
        answer = self.answers.pop(0)
        self.output.append(f"{question}{answer}")
        return answer

    def _record(self, *args) -> None:
        self.output.append(" ".join(str(a) for a in args))

    @property
    def text_output(self) -> str:
        return "\n".join(self.output)


@pytest.fixture
def universe():
    return make_toy_universe(n_frames=5)


# --------------------------------------------------------------------------- #
# selection resolution
# --------------------------------------------------------------------------- #
def test_resid_at_atom_syntax(universe):
    resolver = AtomResolver(universe, ScriptedPrompt([]))
    resolved = resolver.resolve("1@OG")
    assert resolved.selection == "resid 1 and name OG"
    assert resolved.n_atoms == 1
    assert resolved.alias == "SER1_OG"          # alias from what matched
    assert "SER1:OG" in resolved.description


def test_resname_and_number_is_checked(universe):
    resolver = AtomResolver(universe, ScriptedPrompt([]))
    assert resolver.resolve("SER1@OG").n_atoms == 1
    with pytest.raises(MDInteractionsError, match="es SER, no ARG"):
        resolver.resolve("ARG1@OG")


def test_resname_only_and_multiple_atoms(universe):
    resolver = AtomResolver(universe, ScriptedPrompt([]))
    assert resolver.resolve("LIG@C1").selection == "resname LIG and name C1"
    group = resolver.resolve("3@OD1,OD2")
    assert group.n_atoms == 2
    assert group.alias == "ASP3_OD1_OD2"


def test_raw_mdanalysis_selection_still_works(universe):
    resolver = AtomResolver(universe, ScriptedPrompt([]))
    resolved = resolver.resolve("resname WAT and name O")
    assert resolved.n_atoms == 4


def test_unknown_selection_is_rejected(universe):
    resolver = AtomResolver(universe, ScriptedPrompt([]))
    with pytest.raises(MDInteractionsError, match="no casa ningún átomo"):
        resolver.resolve("999@XX")
    with pytest.raises(MDInteractionsError, match="Formato esperado"):
        resolver.resolve("@OG")                 # missing residue part
    with pytest.raises(MDInteractionsError, match="No entiendo el residuo"):
        resolver.resolve("-@OG")                # neither a number nor a name


def test_ask_retries_until_valid_and_reports_atom_count(universe):
    prompt = ScriptedPrompt(["999@XX", "resname WAT and name O", "1@OG"])
    resolver = AtomResolver(universe, prompt)
    resolved = resolver.ask("Átomo")           # rejects the first two
    assert resolved.alias == "SER1_OG"
    assert "no casa ningún átomo" in prompt.text_output
    assert "hace falta exactamente 1" in prompt.text_output


def test_query_helpers_do_not_consume_the_answer(universe):
    prompt = ScriptedPrompt(["?SER", "?1", "?ligandos", "1@OG"])
    resolver = AtomResolver(universe, prompt)
    resolved = resolver.ask("Átomo")
    assert resolved.alias == "SER1_OG"
    out = prompt.text_output
    assert "SER: 1 residuo" in out
    assert "resid 1 = SER" in out
    assert "LIG" in out                        # listed as non-protein residue


# --------------------------------------------------------------------------- #
# full run
# --------------------------------------------------------------------------- #
def test_wizard_writes_a_loadable_config(tmp_path, toy_files):
    topology, trajectory = toy_files
    destination = tmp_path / "generated.yaml"
    answers = [
        str(trajectory),      # trajectory
        "ns", "0.01", "1",    # time unit, dt, stride
        # distance
        "1", "1", "1@OG", "LIG@C1", "d_nuc", "3.2",
        # angle
        "2", "1@OG", "LIG@C1", "LIG@O1", "a_attack",
        # hydrogen bond
        "4", "1@OG", "2@NE2", "hb_ser_his", "n",
        # 2D map needs two observables
        "10", "d_nuc", "a_attack", "fes_test", "s",
        "f",                                    # finish
        "results", "png", "150", "s",           # output
    ]
    prompt = ScriptedPrompt(answers)
    path = run_wizard(topology=topology, output=destination,
                      input_fn=prompt._next, print_fn=prompt._record, force=True)

    assert path == destination
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["system"]["trajectory"] == [str(trajectory)]
    assert raw["system"]["time"] == {"dt": 0.01, "unit": "ns"}

    config = load_config(path)                  # the real parser accepts it
    assert [d.name for d in config.distances.pairs] == ["d_nuc"]
    assert config.distances.pairs[0].threshold == 3.2
    assert [a.name for a in config.angles.angles] == ["a_attack"]
    assert [h.name for h in config.hbonds.pairs] == ["hb_ser_his"]
    assert [m.name for m in config.free_energy.maps] == ["fes_test"]
    assert config.output.formats == ["png"]


def test_wizard_declares_replicas_for_several_trajectories(tmp_path, toy_files):
    topology, trajectory = toy_files
    prompt = ScriptedPrompt([
        "s",                                    # yes, they are replicas
        "ns", "", "1",
        "f",                                    # no analyses
        "results", "png", "150", "n",
    ])
    path = run_wizard(topology=topology, trajectory=[trajectory, trajectory],
                      output=tmp_path / "rep.yaml",
                      input_fn=prompt._next, print_fn=prompt._record, force=True)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert list(raw["system"]["replicas"]) == ["rep1", "rep2"]
    assert Config.from_dict(raw).system.replicas.keys() == {"rep1", "rep2"}


def test_map_requires_two_observables(tmp_path, toy_files):
    topology, _trajectory = toy_files
    prompt = ScriptedPrompt([
        "", "ns", "", "1",
        "10",                                   # 2D map with nothing defined yet
        "f",
        "results", "png", "150", "n",
    ])
    run_wizard(topology=topology, output=tmp_path / "empty.yaml",
               input_fn=prompt._next, print_fn=prompt._record, force=True)
    assert "Necesitas al menos dos observables" in prompt.text_output


def test_missing_topology_is_reported(tmp_path):
    prompt = ScriptedPrompt([])
    with pytest.raises(MDInteractionsError, match="No encuentro la topología"):
        run_wizard(topology=tmp_path / "nope.prmtop", output=tmp_path / "x.yaml",
                   input_fn=prompt._next, print_fn=prompt._record)
