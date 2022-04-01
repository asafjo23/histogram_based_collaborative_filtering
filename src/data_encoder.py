import numpy as np
from pandas import Series, DataFrame


class ProcColumn:
    def __init__(self, column: Series):
        uniq = column.unique()
        self._name_2_index = {o: i for i, o in enumerate(uniq)}
        self._idx_2_name = {i: e for i, e in enumerate(self._name_2_index.keys())}

        self.encoded_col = np.array([self._name_2_index[x] for x in column])

    def get_index(self, name: str) -> int:
        return self._name_2_index[name]

    def get_name(self, index: int) -> str:
        return self._idx_2_name[index]


class DataEncoder:
    def __init__(self, original_df: DataFrame):
        self._user_original_id_to_encoded_id = ProcColumn(original_df.user_id)
        self._item_original_id_to_encoded_id = ProcColumn(original_df.item_id)
        self._encoded_df = original_df.copy()
        self._encoded_df.user_id = self._user_original_id_to_encoded_id.encoded_col
        self._encoded_df.item_id = self._item_original_id_to_encoded_id.encoded_col

    def get_encoded_dataframe(self) -> DataFrame:
        return self._encoded_df

    def get_original_user_id(self, encoded_id: int) -> str:
        return self._user_original_id_to_encoded_id.get_name(index=encoded_id)

    def get_original_item_id(self, encoded_id: int) -> str:
        return self._item_original_id_to_encoded_id.get_name(index=encoded_id)

    def get_encoded_user_id(self, original_id: str) -> int:
        return self._user_original_id_to_encoded_id.get_index(name=original_id)

    def get_encoded_item_id(self, original_id: str) -> int:
        return self._item_original_id_to_encoded_id.get_index(name=original_id)
