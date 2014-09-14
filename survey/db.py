__author__ = 'ilya'


class DB(object):

    def __init__(self, driver=None, host=None, port=None, db=None, user=None,
                 password=None):
        self._driver = driver
        self._connection = self._driver.connect(database=db, user=user,
                                                password=password, host=host,
                                                port=port)
        self._cursor = self._connection.cursor()

    def commit(self):
        self._connection.commit()

    def insert(self, table=None, columns=None, values=None):
        """
        Insert into table values in columns.
        """
        assert(len(values) == len(columns))

        columns_str = ' ('
        for column in columns:
            columns_str += column + ', '
        columns_str = columns_str[:-2]
        columns_str += ') '

        values_str = ' ('
        for value in columns:
            values_str += '%s' + ', '
        values_str = values_str[:-2]
        values_str += ') '

        self._cursor.execute('INSERT INTO ' + table + columns_str + ' VALUES' +
                             values_str, tuple(values))

    # TODO: Implement JOIN
    def select(self, table=None, columns=None, where_constraints=None,
               verbose=False):
        """
        Select specified columns from table where other columns equal to some
        specified values.

        :param columns:
            List of column names to retrive.

        :param where_constraints (optional):
            List of tuples (column name, operator, value,) or ``None``.
        """

        columns_str = ''
        if columns:
            for column in columns:
                columns_str += column + ', '
            columns_str = columns_str[:-2]
        else:
            columns_str = '*'

        if where_constraints:
            where_block = ' '
            for constraint in where_constraints:
                where_block += constraint[0] + ' '
                where_block += constraint[1] + ' '
                where_block += constraint[2] + ' '

            self._cursor.execute('SELECT ' + columns_str + ' FROM ' + table +
                                 'WHERE ' + where_block)
        else:
            self._cursor.execute('SELECT ' + columns_str + ' FROM ' + table)

        rows = self._cursor.fetchall()

        if verbose:
            for row in rows:
                for i, column in enumerate(columns):
                    print columns[i] + ' = ', row[i], '\n'

        return rows

    def update(self, table=None, columns=None, values=None,
               where_constraints=None):
        pass

    def delete(self, table=None, where_constraints=None):
        """
        Delete rows from table where some specified columns equal to some
        specified values.
        """
        if where_constraints:

            where_block = ''
            for constrain in where_constraints:
                where_block += constrain[0] + ' ' + constrain[1] + ' ' + \
                               constrain[2] + ' AND '
            where_block = where_block[:-5]
            where_block += ';'

            self._cursor.execute('DELETE FROM ' + table + ' WHERE ' +
                                 where_block)
            self._connection.commit()

        elif not where_constraints:

            print 'Deleting all from ' + table
            self._cursor.execute('DELETE ALL FROM ' + table)
            self._connection.commit()
